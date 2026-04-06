def test_loop():
    B = 2
    prompt_lens = [10, 20]
    max_new_tokens = 128
    T = max(prompt_lens) + max_new_tokens # 148
    eos_id = 99
    import torch
    x = torch.full((B, T), eos_id, dtype=torch.long)
    for i in range(B):
        x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = 88
    
    all_done = True
    for j in range(B):
        gen_part = x[j, prompt_lens[j]:]
        term_mask = (gen_part == eos_id)
        term_indices = term_mask.nonzero(as_tuple=True)[0]
        if term_indices.numel() > 0:
            print(f"Seq {j} has eos_id at", term_indices[0].item())
        else:
            all_done = False
    print("all_done:", all_done)

test_loop()
