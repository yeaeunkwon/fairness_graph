REASONPROMPTV3 = """<USER>: You are a professional content moderator specializing in hate speech detection. Your task is to analyze whether a given message contains hate speech and explain the specific reasons why it is considered hateful. Please focus on identifying key hateful elements like derogatory language, dehumanization, discrimination, or inflammatory rhetoric. Here are 2 examples:

Example 1:
<USER>: Context 1: The Democrat party is like a dog sniffing its own tail. It stinks.
<ASSISTANT>: Analysis 1: This message contains hate speech for two key reasons:
1. It uses dehumanizing language by comparing a political party to a dog in a demeaning way
2. The phrase "it stinks" expresses explicit contempt and hostility rather than legitimate criticism
These elements combine to promote hatred and division rather than constructive political discourse.

Example 2:
<USER>: Context 2: i agree it is a very high chance this is a white person saying it.
<ASSISTANT>: This message does NOT contain hate speech for these specific reasons:
1. It uses no slurs or derogatory language targeting any group, only referencing "white person" in a neutral, descriptive way
2. The phrase "high chance this is a white person saying it" expresses speculation about who spoke, not contempt or hostility toward a group.
These elements includes a speculative comment rather than incitement of hatred or division.


<USER>: New Context: {context}
<ASSISTANT>: Analysis: """


#no reasoning
TRIPLETPROMPTV4 = '''<USER>:  You are a helpful AI assistant specialized in extracting relationship triplets from text. Your task is to identify the factual relationships stated in a given context, regardless of whether the context is hateful or neutral.

Instructions:
1. Output Format: (head, relation, tail)
2. Guidelines:
   - Use the original words from the text for head, relation, and tail whenever possible.
   - The relation is the predicate (verb or verb phrase) connecting head and tail as stated in the text.
   - Extract the main action relationships between entities (who does what to whom).
   - Break noun phrases into their core noun plus modifiers so entities can be shared as nodes.
     e.g., "persecution of christians" → (persecution, targets, christians)
   - Decompose descriptive phrases into attribute triplets using "is/are". Keep adjacent modifiers together.
     e.g., "stupid clueless parents" → (parents, are, stupid clueless)
   - Maximize connectivity: link distinct entities that appear together using "co-occur" so the graph stays connected.
   - Reuse the same node label for the same entity across triplets (so nodes connect through shared entities).
   - Do not add interpretation or judgment; only extract relationships grounded in the context.

Examples:

Example 1:
Context: "the persecution of christians isn't as bad as the persecution of muslims."
Output:
(persecution, targets, christians)
(persecution, targets, muslims)
(persecution of christians, is not as bad as, persecution of muslims)
(christians, co-occur, muslims)

Example 2:
Context: "just shows how many retarded jerks we have living here buying everything he has. yea, that means your kids you stupid clueless parents. wise up."
Output:
(jerks, are, retarded)
(jerks, living, here)
(jerks, buying, everything)
(parents, are, stupid clueless)
(parents, have, kids)
(jerks, co-occur, parents)
(jerks, co-occur, kids)

Now, please analyze this input:
Context: {context}

<ASSISTANT>: Output:'''
