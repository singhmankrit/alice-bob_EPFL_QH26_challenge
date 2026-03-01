Code for Team: Decide Name Later

This contains the **1-challenge.ipynb** file containing our submission for Parts 1-4

Under the **\code** folder, we have the code for the main challenge, which consists of:

1. Files starting with **bicycle** compares the two Bicycle LDPC Codes [16, 8, 4] and [24, 12, 6] across 3 decoders (PyMatching, Belief Propagation (BP), No QEC).
2. Files starting with **concat3** compares Concatenation Codes where the inner code is a 3-bit repetition code and the outer codes are varying distance repetition codes.
3. Files starting with **repetition** compares the basic repetition code.
4. The file **compare.ipynb** plots and compares 2 things:
    a. Decoders for the same QEC codes
    b. QEC codes for the same decoders
5. The file **ldpc_decoder.py** it is required to run Belief Propagation.
