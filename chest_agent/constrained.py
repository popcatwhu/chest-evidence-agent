"""Initialize JSON string alphabets before the parser caches them."""


def schema_prefix_function(tokenizer_data, json_schema):
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    # Setting parser.config afterwards does not update the cached string alphabet.
    # This must precede JsonSchemaParser construction for Chinese string values.
    parser_config = CharacterLevelParserConfig(
        alphabet=tokenizer_data.tokenizer_alphabet
    )
    parser = JsonSchemaParser(json_schema, config=parser_config)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
