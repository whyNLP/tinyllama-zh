"""
This script will remove the embedding of a model, include the final projection.
"""

from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    input: str = field(
        default=None,
        metadata={
            "help": (
                "The input model path."
            )
        },
    )

    output: str = field(
        default=None,
        metadata={
            "help": (
                "The output model path."
            )
        },
    )
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments,))
    (model_args, ) = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.input)
    assert config.model_type == "llama"

    tokenizer = AutoTokenizer.from_pretrained(model_args.input)

    model = AutoModelForCausalLM.from_pretrained(model_args.input, config=config)
    
    del model.model.embed_tokens
    del model.lm_head

    model.save_pretrained(model_args.output)
    tokenizer.save_pretrained(model_args.output)
    config.save_pretrained(model_args.output)

if __name__ == '__main__':
    main()