"""Initialize JSON string alphabets before the parser caches them."""


def schema_prefix_function(tokenizer_data, json_schema):
    from lmformatenforcer import (
        JsonSchemaParser,
        CharacterLevelParserConfig,
        RegexParser,
    )
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    # Setting parser.config afterwards does not update the cached string alphabet.
    # This must precede JsonSchemaParser construction for Chinese string values.
    parser_config = CharacterLevelParserConfig(
        alphabet=tokenizer_data.tokenizer_alphabet
    )
    parser = JsonSchemaParser(json_schema, config=parser_config)
    # The library's context declares a shared regex cache; lazily constructed
    # child parsers can also inherit a default ASCII alphabet. Build an isolated
    # cache with this model's alphabet before any string field is decoded.
    parser.context.regex_parser_cache = {}

    def prepare_patterns(value):
        if isinstance(value, dict):
            pattern = value.get("pattern")
            if pattern:
                key = pattern[1:] if pattern.startswith("^") else pattern
                key = key[:-1] if key.endswith("$") else key
                parser.context.regex_parser_cache[key] = RegexParser(key, parser_config)
            for child in value.values():
                prepare_patterns(child)
        elif isinstance(value, list):
            for child in value:
                prepare_patterns(child)

    prepare_patterns(json_schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
