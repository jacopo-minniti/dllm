def test_loop():
    B = 1
    prompt_lens = [10]
    T = 10 + 128
    eos_id = 99
    max_new_tokens = 128
    import torch
    x = torch.full((B, T), eos_id, dtype=torch.long)
    for i, p in enumerate([torch.zeros(10)]):
        x[i, :prompt_lens[i]] = p
        x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = 88
    
    gen_part = x[0, prompt_lens[0]:]
    print("gen_part length:", len(gen_part))
    print("gen_part contains eos?", (gen_part == eos_id).any().item())

test_loop()
