"""Small LM Format Enforcer bridge without Transformers-private imports.

Uses the token-enforcer core interface documented in the upstream integration:
https://github.com/noamgat/lm-format-enforcer/blob/main/lmformatenforcer/integrations/transformers.py
"""


def regular_tokens(tokenizer):
    anchor=tokenizer.encode('0',add_special_tokens=False)[-1]
    special=set(tokenizer.all_special_ids)
    for index in range(len(tokenizer)):
        if index in special:
            continue
        contextual=tokenizer.decode([anchor,index])[1:]
        isolated=tokenizer.decode([index])
        yield index,contextual,len(contextual)>len(isolated)


def tokenizer_data(tokenizer):
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    return TokenEnforcerTokenizerData(list(regular_tokens(tokenizer)),
        lambda tokens:tokenizer.decode(tokens).rstrip('\ufffd'),
        tokenizer.eos_token_id,False,len(tokenizer))


def prefix_filter(data,schema):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer
    enforcer=TokenEnforcer(data,JsonSchemaParser(schema))
    return lambda batch_id,sent:enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
