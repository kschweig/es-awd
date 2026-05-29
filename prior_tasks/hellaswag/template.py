SYSTEM_MESSAGE = (
    "You are given a situation followed by four possible endings. "
    "Choose the most appropriate ending by selecting the corresponding number. "
    "Reply with only the option number: 0, 1, 2, or 3."
)

USER_TEMPLATE = (
    "Activity: {activity_label}\n"
    "Context: {context}\n\n"
    "Possible endings:\n"
    "0. {ending_0}\n"
    "1. {ending_1}\n"
    "2. {ending_2}\n"
    "3. {ending_3}\n\n"
    "Which ending best completes the scenario?"
)

