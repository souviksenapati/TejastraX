# Matches key clauses in policy documents
def match_clauses(text, clauses):
    matched = {}
    for clause in clauses:
        if clause.lower() in text.lower():
            matched[clause] = True
        else:
            matched[clause] = False
    return matched
