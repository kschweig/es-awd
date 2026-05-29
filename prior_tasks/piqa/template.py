SYSTEM_MESSAGE = (
    "You are given a physical commonsense goal and two possible solutions. "
    "Choose the better solution by selecting the corresponding number. "
    "Reply with only the option number: 1 or 2."
)

USER_TEMPLATE = (
    "Goal: {goal}\n\n"
    "Possible solutions:\n"
    "1. {solution_1}\n"
    "2. {solution_2}\n\n"
    "Which solution is more likely to work?"
)
