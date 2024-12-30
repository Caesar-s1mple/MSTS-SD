import argparse
import torch
from pathlib import Path
from runs import ar, ar_compile, ss, ss_compile
from models.utils import set_seed

default_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
set_seed(20010101)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 常规设置
    parser.add_argument('--prompt', type=str, default="Hello, my name is")
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--top_p', type=float, default=0.)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--use_cache', type=bool, default=True)
    # 模型指定
    parser.add_argument('--config_path', type=str, default='./models/config/llama-2-7b.json')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/llama-7b/convert/model.pth')
    parser.add_argument('--quantize', type=str, default=None, choices=[None, 'int8', 'int4'])
    parser.add_argument('--dialogue', type=bool, default=False)
    parser.add_argument('--system_prompt', type=str, default='You are a helpful assistant.')
    # 采样策略
    parser.add_argument('--sample_strategy', type=str, default='ss', choices=['ar', 'ss'])
    # 如果使用ss，则需设置以下draft model
    parser.add_argument('--draft_config_path', type=str, default='./models/config/llama-2-160m.json')
    parser.add_argument('--draft_checkpoint_path', type=str, default='./checkpoints/llama-160m/convert/model.pth')
    parser.add_argument('--draft_quantize', type=str, default=None, choices=[None, 'int8', 'int4'])
    parser.add_argument('--gamma', type=int, default=7, help='draft tokens个数')

    args = parser.parse_args()

    if args.sample_strategy == 'ar':
        ar.main(prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                config_path=Path(args.config_path),
                checkpoint_path=Path(args.checkpoint_path),
                quantize=args.quantize,
                use_cache=args.use_cache,
                dialogue=args.dialogue,
                system_prompt=args.system_prompt
                )
    elif args.sample_strategy == 'ss':
        ss.main(prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                config_path=Path(args.config_path),
                checkpoint_path=Path(args.checkpoint_path),
                draft_config_path=Path(args.draft_config_path),
                draft_checkpoint_path=Path(args.draft_checkpoint_path),
                quantize=args.quantize,
                draft_quantize=args.draft_quantize,
                gamma=args.gamma,
                use_cache=args.use_cache,
                dialogue=args.dialogue,
                system_prompt=args.system_prompt
                )
