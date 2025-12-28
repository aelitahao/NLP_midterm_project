"""Decoding strategy module - Greedy decoding and Beam Search"""

from typing import List
import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(model, src, src_lengths=None, max_len=100, sos_idx=1, eos_idx=2, model_type='rnn'):
    """Greedy decoding"""
    model.eval()
    device = src.device
    batch_size = src.size(0)

    if model_type == 'rnn':
        enc_out, hidden = model.encode(src, src_lengths)
        mask = model.create_mask(src)
        tokens = [torch.full((batch_size,), sos_idx, device=device)]
        context = None

        for _ in range(max_len):
            out, hidden, context, _ = model.decode_step(tokens[-1], hidden, enc_out, mask, context)
            next_tok = out.argmax(dim=-1)
            tokens.append(next_tok)
            if (next_tok == eos_idx).all():
                break

        return torch.stack(tokens[1:], dim=1)
    else:
        src_mask = model.create_src_mask(src)
        enc_out = model.encode(src, src_mask)
        tgt = torch.full((batch_size, 1), sos_idx, device=device)

        for _ in range(max_len):
            out = model.decode(tgt, enc_out, src_mask=src_mask)
            next_tok = out[:, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if (next_tok.squeeze(-1) == eos_idx).all():
                break

        return tgt[:, 1:]


@torch.no_grad()
def beam_search(model, src, src_lengths=None, beam_width=5, max_len=100,
                sos_idx=1, eos_idx=2, model_type='rnn') -> List[int]:
    """Beam Search decoding (single sample)"""
    model.eval()
    device = src.device

    if model_type == 'rnn':
        enc_out, hidden = model.encode(src, src_lengths)
        mask = model.create_mask(src)

        beams = [(0.0, [sos_idx], hidden, None)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, tokens, h, ctx in beams:
                if tokens[-1] == eos_idx:
                    completed.append((score, tokens))
                    continue

                inp = torch.tensor([tokens[-1]], device=device)
                out, new_h, new_ctx, _ = model.decode_step(inp, h, enc_out, mask, ctx)
                log_probs = F.log_softmax(out, dim=-1).squeeze(0)
                topk = log_probs.topk(beam_width)

                for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((score + prob, tokens + [idx], new_h, new_ctx))

            if not candidates:
                break
            beams = sorted(candidates, key=lambda x: -x[0])[:beam_width]

        completed.extend([(s, t) for s, t, _, _ in beams])
        best = max(completed, key=lambda x: x[0] / len(x[1]))
        result = best[1][1:]  # Remove sos
        return result[:-1] if result and result[-1] == eos_idx else result

    else:
        src_mask = model.create_src_mask(src)
        enc_out = model.encode(src, src_mask).expand(beam_width, -1, -1)
        src_mask = src_mask.expand(beam_width, -1, -1, -1)

        tgt = torch.full((beam_width, 1), sos_idx, device=device)
        scores = torch.zeros(beam_width, device=device)
        done = torch.zeros(beam_width, dtype=torch.bool, device=device)

        for _ in range(max_len):
            out = model.decode(tgt, enc_out, src_mask=src_mask)
            log_probs = F.log_softmax(out[:, -1], dim=-1)
            log_probs[done] = float('-inf')
            log_probs[done, 0] = 0

            vocab = log_probs.size(-1)
            cand_scores = (scores.unsqueeze(1) + log_probs).view(-1)
            topk = cand_scores.topk(beam_width)

            beam_idx = topk.indices // vocab
            tok_idx = topk.indices % vocab

            tgt = torch.cat([tgt[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores = topk.values
            done = done[beam_idx] | (tok_idx == eos_idx)

            if done.all():
                break

        best = scores.argmax()
        result = tgt[best, 1:].tolist()
        return result[:result.index(eos_idx)] if eos_idx in result else result
