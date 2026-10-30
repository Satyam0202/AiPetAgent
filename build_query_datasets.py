import csv
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT / "data"
ORIGIN="synthetic_template_augmented"
DEPARTMENTS=("IT", "HR", "Finance", "Sales", "Marketing", "Operations")
NAMES=("Aarav Shah", "Maya Patel", "Rohan Mehta", "Isha Verma", "Kabir Rao")
TRAIN_FRAMES=(
    "please",
    "for me",
    "from the employee database",
    "for a company report",
    "in our company",
    "from the records",
    "for the team",
    "in the database",
    "right now",
    "as a report",
)
TEST_FRAMES=(
    "for my report",
    "from our company records",
    "for review",
    "in the employee table",
    "for the company",
)
ROWS=[]


def add_rows(label, train_questions, test_questions):
    wrapper_sets={
        "train": (
            "Please {question}",
            "Could you {question}",
            "I need you to {question}",
            "For a report, {question}",
            "I would like to {question}",
            "Can you {question}",
            "Request: {question}",
            "Help me {question}",
        ),
        "test": (
            "Would you {question}",
            "I want to {question}",
            "For review, {question}",
            "Kindly {question}",
            "Please {question}",
            "Could you {question}",
            "I need you to {question}",
            "For a report, {question}",
            "I would like to {question}",
            "Can you {question}",
        ),
    }
    rng=random.Random(label)
    for split, source_questions, target in (
        ("train", train_questions, 400),
        ("test", test_questions, 100),
    ):
        base_questions=list(dict.fromkeys(q.strip() for q in source_questions if q.strip()))
        wraps=wrapper_sets[split]
        augmented=[
            template.format(question=question)
            for template in wraps
            for question in base_questions
        ]
        augmented=list(dict.fromkeys(question for question in augmented if question not in base_questions))
        if len(base_questions) > target:
            rng.shuffle(base_questions)
            questions=base_questions[:target]
        else:
            rng.shuffle(augmented)
            questions=base_questions + augmented[:target - len(base_questions)]
        for question in questions:
            ROWS.append({
                "question": question.strip(),
                "intent": label,
                "split": split,
                "data_origin": ORIGIN,
            })


def with_frames(cores, frames):
    return [f"{core} {frame}" for core in cores for frame in frames]


def make_intent_rows():
    subject=("employees", "workers", "staff", "employee records", "the workforce")
    all_actions=(
        "show all", "list all", "display all", "retrieve all",
        "give me all", "get all", "fetch all", "view all",
    )
    add_rows(
        "SELECT_ALL",
        with_frames([f"{action} {who}" for action in all_actions[:6] for who in subject], TRAIN_FRAMES),
        with_frames([f"{action} {who}" for action in all_actions[6:] for who in subject], TEST_FRAMES),
    )

    dept_actions=(
        "show", "list", "find", "display", "get", "retrieve", "search for", "fetch",
    )
    dept_cores=[
        f"{action} employees in the {department} department"
        for action in dept_actions for department in DEPARTMENTS
    ]
    add_rows(
        "FILTER_BY_DEPARTMENT",
        with_frames(dept_cores[:len(dept_cores) * 3 // 4], TRAIN_FRAMES),
        with_frames(dept_cores[len(dept_cores) * 3 // 4:], TEST_FRAMES),
    )

    comparisons=(
        ("greater than", ">"), ("above", ">"), ("over", ">"),
        ("at least", ">="), ("less than", "<"), ("below", "<"),
        ("under", "<"), ("at most", "<="),
        ("equal to", "="), ("exactly", "="),
    )
    salary_actions=("find", "show", "list", "get", "display", "search for", "retrieve", "fetch")
    salary_train=[
        f"{action} employees earning {word} {value}"
        for action in salary_actions[:6]
        for word, _ in comparisons
        for value in (30000, 45000, 60000, 75000, 90000)
    ]
    salary_test=[
        f"{action} employees with salary {word} {value}"
        for action in salary_actions[6:]
        for word, _ in comparisons
        for value in (35000, 50000, 65000, 80000, 95000)
    ]
    add_rows("FILTER_BY_SALARY", salary_train, salary_test)

    age_actions=("find", "show", "list", "get", "display", "search for", "retrieve", "fetch")
    age_train=[
        f"{action} employees {word} {value} years old"
        for action in age_actions[:6]
        for word in ("older than", "over", "above", "at least", "younger than", "under", "below", "at most", "equal to", "exactly")
        for value in (22, 25, 30, 35, 40)
    ]
    age_test=[
        f"{action} workers {word} {value} years old"
        for action in age_actions[6:]
        for word in ("older than", "over", "above", "at least", "younger than", "under", "below", "at most", "equal to", "exactly")
        for value in (23, 27, 32, 37, 42)
    ]
    add_rows("FILTER_BY_AGE", age_train, age_test)

    count_actions=("count", "how many", "give me the number of", "calculate the number of",
                   "tell me how many", "what is the count of", "get a count of", "find the number of")
    count_objects=("employees", "workers", "staff members", "people", "team members")
    count_train=[f"{action} {who}" for action in count_actions[:6] for who in count_objects]
    count_test=[f"{action} {who}" for action in count_actions[6:] for who in count_objects]
    add_rows("COUNT_EMPLOYEES", with_frames(count_train, TRAIN_FRAMES),
             with_frames(count_test, TEST_FRAMES))

    for label, measures, actions in (
        ("AVERAGE_SALARY", ("average salary", "mean salary", "average pay", "mean pay", "average employee earnings"),
         ("calculate", "find", "show", "get", "display", "tell me", "work out", "report")),
        ("MAXIMUM_SALARY", ("highest salary", "maximum salary", "largest pay", "top salary", "highest employee earnings"),
         ("find", "show", "get", "display", "tell me", "identify", "report", "retrieve")),
        ("MINIMUM_SALARY", ("lowest salary", "minimum salary", "smallest pay", "bottom salary", "lowest employee earnings"),
         ("find", "show", "get", "display", "tell me", "identify", "report", "retrieve")),
    ):
        cores=[f"{action} the {measure}" for action in actions for measure in measures]
        train_cores=cores[:len(cores) * 3 // 4]
        test_cores=cores[len(cores) * 3 // 4:]
        train_questions=with_frames(train_cores, TRAIN_FRAMES)
        if label == "AVERAGE_SALARY":
            train_questions.extend(("What is the average salary?", "What is average pay?"))
        add_rows(label, train_questions, with_frames(test_cores, TEST_FRAMES))

    sort_train=(
        "sort employees by salary ascending", "order employees by salary from low to high",
        "sort employees by salary from lowest to highest",
        "list employees starting with the lowest salary", "rank employees by salary ascending",
        "show employees ordered by salary in ascending order",
    )
    sort_test=(
        "arrange workers by salary from smallest to largest",
        "display employees in salary ascending order",
    )
    add_rows("ORDER_BY_SALARY_ASC", with_frames(sort_train, TRAIN_FRAMES),
             with_frames(sort_test, TEST_FRAMES))

    sort_train=(
        "sort employees by salary descending", "order employees by salary from high to low",
        "sort employees by salary from highest to lowest",
        "order employees from highest paid to lowest paid",
        "list employees starting with the highest salary", "rank employees by salary descending",
        "show employees ordered by salary in descending order",
    )
    sort_test=(
        "arrange workers by salary from largest to smallest",
        "display employees in salary descending order",
    )
    add_rows("ORDER_BY_SALARY_DESC", with_frames(sort_train, TRAIN_FRAMES),
             with_frames(sort_test, TEST_FRAMES))

    group_train=(
        "count employees in each department", "show employee count by department",
        "group employees by department and count them", "report the number of staff per department",
        "give me a department wise employee count", "summarize employees for each department",
    )
    group_test=(
        "show how many workers belong to every department",
        "provide a count of staff grouped by department",
    )
    add_rows("GROUP_BY_DEPARTMENT", with_frames(group_train, TRAIN_FRAMES),
             with_frames(group_test, TEST_FRAMES))

    insert_train=(
        "insert employee", "add employee", "create employee", "add a new employee",
        "insert a new employee", "create a new employee record",
    )
    insert_test=("register employee", "add a staff member")
    insert_formats=(
        "id {id}, name {name}, department {department}, salary {salary}, age {age}",
        "with id {id}, name {name}, department {department}, salary {salary}, and age {age}",
        "record: id {id}; name {name}; department {department}; salary {salary}; age {age}",
        "employee id {id}, name {name}, department {department}, salary {salary}, age {age}",
    )
    values=[
        {"id": 501 + i, "name": name, "department": DEPARTMENTS[i % len(DEPARTMENTS)],
         "salary": 35000 + i * 1000, "age": 22 + i % 30}
        for i, name in enumerate(NAMES)
    ]
    insert_train_q=[
        f"{action} {fmt.format(**value)}"
        for action in insert_train for fmt in insert_formats for value in values
    ]
    insert_test_q=[
        f"{action} {fmt.format(**value)}"
        for action in insert_test for fmt in insert_formats for value in values
    ]
    add_rows("INSERT_EMPLOYEE", with_frames(insert_train_q, TRAIN_FRAMES),
             with_frames(insert_test_q, TEST_FRAMES))

    for label, actions, values_by_action in (
        ("UPDATE_SALARY", ("update salary", "change salary", "set salary", "revise salary",
                           "adjust salary", "modify salary", "raise salary", "change the pay of"),
         (40000, 50000, 60000, 70000, 80000)),
        ("UPDATE_AGE", ("update age", "change age", "set age", "revise age",
                        "adjust age", "modify age", "correct age", "change the age of"),
         (24, 28, 32, 36, 40)),
    ):
        train_q=[
            f"{action} for employee id {501 + employee_id} to {value}"
            for action in actions[:6] for employee_id in range(10) for value in values_by_action
        ]
        test_q=[
            f"{action} of employee id {520 + employee_id} to {value}"
            for action in actions[6:] for employee_id in range(10) for value in values_by_action
        ]
        add_rows(label, with_frames(train_q, TRAIN_FRAMES), with_frames(test_q, TEST_FRAMES))

    dept_update_train=(
        "change department for employee id", "update department for employee id",
        "set department for employee id", "move employee id",
        "transfer employee id", "assign employee id",
    )
    dept_update_test=("switch department for employee id", "change team for employee id")
    update_dept_train=[
        f"{action} {501 + employee_id} to {department}"
        for action in dept_update_train for employee_id in range(10) for department in DEPARTMENTS
    ]
    update_dept_test=[
        f"{action} {520 + employee_id} to {department}"
        for action in dept_update_test for employee_id in range(10) for department in DEPARTMENTS
    ]
    add_rows("UPDATE_DEPARTMENT", with_frames(update_dept_train, TRAIN_FRAMES),
             with_frames(update_dept_test, TEST_FRAMES))

    name_train=("change name", "update name", "set name", "rename employee",
                "correct employee name", "modify employee name")
    name_test=("replace employee name", "edit employee name")
    name_values=("Nisha Singh", "Aditya Kumar", "Sara Ali", "Vikram Joshi", "Tara Das")
    name_train_q=[
        f"{action} for employee id {501 + employee_id} to {name}"
        for action in name_train for employee_id in range(10) for name in name_values
    ]
    name_test_q=[
        f"{action} of employee id {520 + employee_id} to {name}"
        for action in name_test for employee_id in range(10) for name in name_values
    ]
    add_rows("UPDATE_NAME", with_frames(name_train_q, TRAIN_FRAMES),
             with_frames(name_test_q, TEST_FRAMES))

    delete_train=("delete employee", "remove employee", "erase employee record",
                  "delete the record for employee", "remove the record of employee",
                  "drop employee record")
    delete_test=("delete staff member", "remove worker record")
    delete_train_q=[
        f"{action} id {501 + i}" for action in delete_train for i in range(50)
    ]
    delete_test_q=[
        f"{action} id {551 + i}" for action in delete_test for i in range(50)
    ]
    add_rows("DELETE_EMPLOYEE", with_frames(delete_train_q, TRAIN_FRAMES),
             with_frames(delete_test_q, TEST_FRAMES))


AMBIGUOUS={
    "SELECT_ALL": ("show the records", "show everything", "display some information", "list the data"),
    "FILTER_BY_DEPARTMENT": ("show employees in a department", "find a department's employees", "show workers by team", "list staff in a group"),
    "FILTER_BY_SALARY": ("find high salary employees", "show employees by salary", "find employees earning more", "show workers with a salary condition"),
    "FILTER_BY_AGE": ("show older employees", "find employees by age", "list workers by age", "show employees above an age"),
    "COUNT_EMPLOYEES": ("how many employees", "count the staff", "tell me how many", "what is the headcount"),
    "AVERAGE_SALARY": ("calculate the average", "show average pay", "find the mean", "calculate average earnings"),
    "MAXIMUM_SALARY": ("find the highest", "show maximum pay", "find the top earning", "show the largest salary"),
    "MINIMUM_SALARY": ("find the lowest", "show minimum pay", "find the smallest salary", "show the bottom earning"),
    "ORDER_BY_SALARY_ASC": ("sort employees by salary", "order employees by pay", "arrange salary", "list employees in salary order"),
    "ORDER_BY_SALARY_DESC": ("rank employees from highest to lowest", "order top-paid staff", "arrange the highest salaries", "list best-paid employees first"),
    "GROUP_BY_DEPARTMENT": ("group employees", "summarize departments", "count by department", "report department totals"),
    "INSERT_EMPLOYEE": ("add a new employee", "insert employee details", "create an employee", "register a staff member"),
    "UPDATE_SALARY": ("update salary for employee id 20", "change employee pay", "set pay for an employee", "revise employee salary"),
    "UPDATE_DEPARTMENT": ("change department for employee id 20", "move an employee", "assign employee to a team", "transfer a staff member"),
    "UPDATE_AGE": ("update age for employee id 20", "change employee age", "set the age for a worker", "correct staff age"),
    "UPDATE_NAME": ("change name for employee id 20", "rename an employee", "edit staff name", "update the employee name"),
    "DELETE_EMPLOYEE": ("delete an employee", "remove an employee record", "erase an employee", "delete the staff record"),
}


def write_rows(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer=csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    make_intent_rows()
    random.Random(42).shuffle(ROWS)
    write_rows(DATA_DIR / "intent_dataset.csv",
               ("question", "intent", "split", "data_origin"), ROWS)

    clear_rows=[]
    for label in AMBIGUOUS:
        clear_rows.extend(
            {"question": row["question"], "ambiguity": "CLEAR",
             "split": row["split"], "data_origin": ORIGIN}
            for row in ROWS if row["intent"] == label
        )
    clear_by_split={
        split: [row for row in clear_rows if row["split"] == split][::4]
        for split in ("train", "test")
    }
    train_prefixes=(
        "Please", "Could you", "I need you to", "For a report,",
        "Can you", "Help me", "Request:", "Kindly",
        "I would like you to", "I want you to",
    )
    test_prefixes=("Would you", "For review,", "Tell me to", "I want to", "Please help me")
    suffixes=("please", "for a report", "from our records", "for the company", "in the database")
    needs_rows=[]
    for questions in AMBIGUOUS.values():
        for split, examples, prefixes, target in (
            ("train", questions[:2], train_prefixes, 100),
            ("test", questions[2:], test_prefixes, 25),
        ):
            variants=[
                f"{prefix} {question} {suffix}"
                for prefix in prefixes for question in examples for suffix in suffixes
            ]
            variants=list(dict.fromkeys(variants))[:target]
            needs_rows.extend(
                {"question": question, "ambiguity": "NEEDS_CLARIFICATION",
                 "split": split, "data_origin": ORIGIN}
                for question in variants
            )
    ambiguity_rows=clear_by_split["train"] + clear_by_split["test"] + needs_rows
    random.Random(43).shuffle(ambiguity_rows)
    write_rows(DATA_DIR / "ambiguity_dataset.csv",
               ("question", "ambiguity", "split", "data_origin"), ambiguity_rows)
    print(f"Intent rows: {len(ROWS)}")
    print(f"Ambiguity rows: {len(ambiguity_rows)}")
    print("Intent counts by split:")
    counts={}
    for row in ROWS:
        counts.setdefault((row["split"], row["intent"]), 0)
        counts[(row["split"], row["intent"])]+=1
    for (split, intent), count in sorted(counts.items()):
        print(f"{split:5} {intent:24} {count}")


if __name__ == "__main__":
    main()
