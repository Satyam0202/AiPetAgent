import csv
import json
import os
import re
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT / "data"
DEPARTMENTS=("IT", "HR", "Finance", "Sales", "Marketing", "Operations")
INTENT_FILE=DATA_DIR / "intent_dataset.csv"
AMBIGUITY_FILE=DATA_DIR / "ambiguity_dataset.csv"


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def load_training_data(path, label_column):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    questions=[]
    labels=[]
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader=csv.DictReader(file)
        required={"question", label_column, "split"}
        missing=required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["split"].strip().lower() == "train":
                question=(row["question"] or "").strip()
                label=(row[label_column] or "").strip()
                if question and label:
                    questions.append(question)
                    labels.append(label)

    if not questions or len(set(labels)) < 2:
        raise ValueError(f"{path.name} does not contain enough labeled training rows.")
    return questions, labels


def train_models():
    intent_questions, intent_labels=load_training_data(INTENT_FILE, "intent")
    ambiguity_questions, ambiguity_labels=load_training_data(
        AMBIGUITY_FILE, "ambiguity"
    )

    intent_vec=TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        ngram_range=(1, 2),
    )
    ambiguity_vec=TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        ngram_range=(1, 2),
    )
    X_intent=intent_vec.fit_transform(intent_questions)
    X_ambiguity=ambiguity_vec.fit_transform(ambiguity_questions)

    ann=MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=128,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=42,
    )
    rf=RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )
    ann.fit(X_intent, intent_labels)
    rf.fit(X_ambiguity, ambiguity_labels)
    return ann, intent_vec, rf, ambiguity_vec


def department_from(text):
    names="|".join(re.escape(name) for name in sorted(DEPARTMENTS, key=len, reverse=True))
    match=re.search(
        rf"\b(?:in|from|belonging to|belong to|assigned to|working in)"
        rf"\s+(?:the\s+)?({names})\b"
        rf"|\b({names})\s+(?:department|team|group)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match=re.search(
            rf"\b(?:department|team)\s+(?:(?:is|to|=)\s+)?({names})\b"
            rf"|\b(?:to|into)\s+({names})\b",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    value=next(group for group in match.groups() if group is not None)
    return next(name for name in DEPARTMENTS if name.lower() == value.lower())


def department_value_from(text):
    names="|".join(re.escape(name) for name in sorted(DEPARTMENTS, key=len, reverse=True))
    matches=re.findall(rf"\b({names})\b", text, flags=re.IGNORECASE)
    if not matches:
        return None
    value=matches[-1]
    return next(name for name in DEPARTMENTS if name.lower() == value.lower())


def number_from(text):
    match=re.search(r"\b\d[\d,]*\b", text)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def comparison_from(text):
    match=re.search(
        r"(>=|<=|=|>|<)|\b(at least|no less than|greater than or equal to|"
        r"at most|no more than|less than or equal to|more than|greater than|"
        r"equal to|exactly|exceeds?|above|over|less than|below|under|fewer than|older than|"
        r"younger than)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    if match.group(1):
        return match.group(1)
    phrase=match.group(2).lower()
    if phrase in {"at least", "no less than", "greater than or equal to"}:
        return ">="
    if phrase in {"at most", "no more than", "less than or equal to"}:
        return "<="
    if phrase in {"equal to", "exactly"}:
        return "="
    if phrase in {
        "less than", "below", "under", "fewer than", "younger than"
    }:
        return "<"
    return ">"


def subject_is_supported(intent, question):
    text=question.lower()
    words={
        "SELECT_ALL": ("employee", "staff", "worker", "workforce", "people"),
        "FILTER_BY_DEPARTMENT": (
            "employee", "staff", "worker", "people", "department", "team", "group"
        ),
        "FILTER_BY_SALARY": (
            "salary", "salaries", "pay", "paid", "earn", "earning", "earnings"
        ),
        "COUNT_EMPLOYEES": (
            "employee", "staff", "worker", "people", "headcount", "workforce"
        ),
        "AVERAGE_SALARY": ("salary", "salaries", "pay", "earn"),
        "MAXIMUM_SALARY": ("salary", "salaries", "pay", "earn"),
        "MINIMUM_SALARY": ("salary", "salaries", "pay", "earn"),
        "FILTER_BY_AGE": ("age", "aged", "older", "younger", "year"),
        "ORDER_BY_SALARY_ASC": ("employee", "worker", "staff", "salary", "pay"),
        "ORDER_BY_SALARY_DESC": ("employee", "worker", "staff", "salary", "pay"),
        "GROUP_BY_DEPARTMENT": ("employee", "worker", "staff", "department"),
        "INSERT_EMPLOYEE": ("employee", "worker", "staff", "insert", "add", "register"),
        "UPDATE_SALARY": ("employee", "worker", "salary", "pay"),
        "UPDATE_DEPARTMENT": ("employee", "worker", "department", "team"),
        "UPDATE_AGE": ("employee", "worker", "age"),
        "UPDATE_NAME": ("employee", "worker", "name"),
        "DELETE_EMPLOYEE": ("employee", "worker", "staff", "record", "delete", "remove"),
    }
    return any(word in text for word in words.get(intent, ()))


def clarification_for(intent, question):
    text=question.lower()
    if re.search(r"\b(high|low|good|large|small)\s+salary\b", text):
        return "salary", "What salary comparison and number should I use? Example: > 50000"
    if re.search(r"\b(best|top)\s+employees?\b", text):
        return "ranking", "Which available metric should I rank by? Type salary or quit."
    if "recent" in text:
        return "unsupported", "This sample table has no date column, so recent employees cannot be queried."
    if intent == "FILTER_BY_DEPARTMENT" and not department_from(question):
        return "department", "Which department? Choose IT, HR, Finance, Sales, Marketing, or Operations."
    if intent in {"FILTER_BY_SALARY", "FILTER_BY_AGE"}:
        if number_from(question) is None or comparison_from(question) is None:
            field="salary" if intent == "FILTER_BY_SALARY" else "age"
            return field, f"What comparison and {field} value should I use? Example: > 50000"
    if intent == "INSERT_EMPLOYEE" and insert_values_from(question) is None:
        return "insert", (
            "Give all fields like this: id 501, name Asha Patel, department IT, "
            "salary 50000, age 30"
        )
    if intent in {"UPDATE_SALARY", "UPDATE_AGE"}:
        if employee_id_from(question) is None or last_number_from(question) is None:
            field="salary" if intent == "UPDATE_SALARY" else "age"
            return field, f"Give employee id and new {field}, e.g. employee id 12 to 50000"
    if intent == "UPDATE_DEPARTMENT":
        if employee_id_from(question) is None or department_value_from(question) is None:
            return "department_update", (
                "Give employee id and new department, e.g. employee id 12 to IT"
            )
    if intent == "UPDATE_NAME":
        if employee_id_from(question) is None or update_name_from(question) is None:
            return "name_update", (
                "Give employee id and new name, e.g. employee id 12 to Asha Patel"
            )
    if intent == "DELETE_EMPLOYEE" and employee_id_from(question) is None:
        return "delete", "Which employee id should be deleted?"
    return None, ""


def last_number_from(text):
    numbers=re.findall(r"\b\d[\d,]*\b", text)
    if len(numbers) < 2:
        return None
    return int(numbers[-1].replace(",", ""))


def employee_id_from(text):
    match=re.search(
        r"\b(?:employee|worker|staff member)\s+(?:id|number|#)\s*(\d+)\b"
        r"|\b(?:employee|worker)\s+#\s*(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(next(value for value in match.groups() if value is not None))


def insert_values_from(text):
    employee_id=re.search(r"\bid\s*(?:is|=|:)?\s*(\d+)\b", text, flags=re.IGNORECASE)
    salary=re.search(
        r"\b(?:salary|pay)\s*(?:is|=|:)?\s*\$?(\d[\d,]*)\b",
        text,
        flags=re.IGNORECASE,
    )
    age=re.search(r"\bage\s*(?:is|=|:)?\s*(\d+)\b", text, flags=re.IGNORECASE)
    name=re.search(
        r"\bname\s*(?:is|=|:)?\s*(.+?)\s*(?=,|;|\bdepartment\b|\bsalary\b|\bage\b|\bid\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    department=department_from(text)
    if not all((employee_id, salary, age, name, department)):
        return None
    clean_name=name.group(1).strip(" \t,;:'\"")
    if not clean_name or not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,98}", clean_name):
        return None
    values=(
        int(employee_id.group(1)),
        clean_name,
        department,
        int(salary.group(1).replace(",", "")),
        int(age.group(1)),
    )
    if values[0] < 1 or values[3] < 1 or not 18 <= values[4] <= 100:
        return None
    return values


def update_name_from(text):
    match=re.search(
        r"\bto\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2})\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def make_query(intent, question):
    number=number_from(question)
    operator=comparison_from(question)
    department=department_from(question)
    table="company_db.employees"
    if intent == "SELECT_ALL":
        if not subject_is_supported(intent, question):
            raise ValueError("This question is outside the employee database scope.")
        return f"SELECT id, name, department, salary, age FROM {table} LIMIT 100;", (), False
    if intent == "FILTER_BY_DEPARTMENT" and department:
        return (
            f"SELECT id, name, department, salary, age FROM {table} "
            f"WHERE department = '{department}' LIMIT 100;", (), False
        )
    if intent == "FILTER_BY_SALARY" and number is not None and operator:
        return (
            f"SELECT id, name, department, salary, age FROM {table} "
            f"WHERE salary {operator} {number} LIMIT 100;", (), False
        )
    if intent == "FILTER_BY_AGE" and number is not None and operator:
        return (
            f"SELECT id, name, department, salary, age FROM {table} "
            f"WHERE age {operator} {number} LIMIT 100;", (), False
        )
    if intent == "COUNT_EMPLOYEES":
        if not subject_is_supported(intent, question):
            raise ValueError("This question is outside the employee database scope.")
        return f"SELECT COUNT(*) AS employee_count FROM {table};", (), False
    if intent == "AVERAGE_SALARY":
        return f"SELECT AVG(salary) AS average_salary FROM {table};", (), False
    if intent == "MAXIMUM_SALARY":
        return f"SELECT MAX(salary) AS highest_salary FROM {table};", (), False
    if intent == "MINIMUM_SALARY":
        return f"SELECT MIN(salary) AS lowest_salary FROM {table};", (), False
    if intent == "ORDER_BY_SALARY_ASC":
        return (
            f"SELECT id, name, department, salary, age FROM {table} "
            "ORDER BY salary ASC LIMIT 100;", (), False
        )
    if intent == "ORDER_BY_SALARY_DESC":
        return (
            f"SELECT id, name, department, salary, age FROM {table} "
            "ORDER BY salary DESC LIMIT 100;", (), False
        )
    if intent == "GROUP_BY_DEPARTMENT":
        return (
            f"SELECT department, COUNT(*) AS employee_count FROM {table} "
            "GROUP BY department ORDER BY department LIMIT 100;", (), False
        )
    if intent == "INSERT_EMPLOYEE":
        values=insert_values_from(question)
        if values is not None:
            return (
                f"INSERT INTO {table} (id, name, department, salary, age) "
                "VALUES (%s, %s, %s, %s, %s);", values, True
            )
    if intent in {"UPDATE_SALARY", "UPDATE_AGE"}:
        employee_id=employee_id_from(question)
        value=last_number_from(question)
        if employee_id is not None and employee_id > 0 and value is not None:
            field="salary" if intent == "UPDATE_SALARY" else "age"
            if value < 1 or (field == "age" and not 18 <= value <= 100):
                raise ValueError(f"Please provide a valid positive {field} value.")
            return f"UPDATE {table} SET {field} = %s WHERE id = %s;", (value, employee_id), True
    if intent == "UPDATE_DEPARTMENT":
        employee_id=employee_id_from(question)
        new_department=department_value_from(question)
        if employee_id is not None and employee_id > 0 and new_department is not None:
            return (
                f"UPDATE {table} SET department = %s WHERE id = %s;",
                (new_department, employee_id), True
            )
    if intent == "UPDATE_NAME":
        employee_id=employee_id_from(question)
        name=update_name_from(question)
        if employee_id is not None and name and len(name) <= 100:
            return f"UPDATE {table} SET name = %s WHERE id = %s;", (name, employee_id), True
    if intent == "DELETE_EMPLOYEE":
        employee_id=employee_id_from(question)
        if employee_id is not None and employee_id > 0:
            return f"DELETE FROM {table} WHERE id = %s;", (employee_id,), True
    raise ValueError(
        "I could not build a supported query. Include the required employee id and values."
    )


def is_allowed_query(query):
    statement=" ".join(query.strip().split())
    if statement.endswith(";"):
        statement=statement[:-1].strip()
    if not statement or ";" in statement or "--" in statement or "/*" in statement:
        return False

    table=r"company_db\.employees"
    columns=r"id, name, department, salary, age"
    patterns=(
        rf"SELECT {columns} FROM {table} LIMIT 100",
        rf"SELECT {columns} FROM {table} WHERE department = "
        rf"'(?:IT|HR|Finance|Sales|Marketing|Operations)' LIMIT 100",
        rf"SELECT {columns} FROM {table} WHERE salary (?:>=|<=|=|>|<) \d+ LIMIT 100",
        rf"SELECT {columns} FROM {table} WHERE age (?:>=|<=|=|>|<) \d+ LIMIT 100",
        rf"SELECT {columns} FROM {table} ORDER BY salary (?:ASC|DESC) LIMIT 100",
        rf"SELECT department, COUNT\(\*\) AS employee_count FROM {table} "
        rf"GROUP BY department ORDER BY department LIMIT 100",
        rf"SELECT COUNT\(\*\) AS employee_count FROM {table}",
        rf"SELECT AVG\(salary\) AS average_salary FROM {table}",
        rf"SELECT MAX\(salary\) AS highest_salary FROM {table}",
        rf"SELECT MIN\(salary\) AS lowest_salary FROM {table}",
        rf"INSERT INTO {table} \(id, name, department, salary, age\) "
        rf"VALUES \(%s, %s, %s, %s, %s\)",
        rf"UPDATE {table} SET (?:salary|age|department|name) = %s WHERE id = %s",
        rf"DELETE FROM {table} WHERE id = %s",
    )
    return any(re.fullmatch(pattern, statement, flags=re.IGNORECASE) for pattern in patterns)


def connect_mysql():
    from getpass import getpass

    try:
        import mysql.connector
    except ImportError as error:
        raise RuntimeError(
            "MySQL connector is missing. Install it with "
            "`python -m pip install mysql-connector-python`."
        ) from error

    host=os.environ.get("MYSQL_HOST", "127.0.0.1")
    port=int(os.environ.get("MYSQL_PORT", "3306"))
    user=os.environ.get("MYSQL_USER") or input("MySQL username [root]: ").strip() or "root"
    password=getpass("MySQL password (input stays hidden): ")

    try:
        connection=mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="company_db",
            connection_timeout=5,
        )
    except mysql.connector.Error as error:
        raise RuntimeError(
            f"Could not connect to MySQL at {host}:{port} as {user}. "
            "Check that MySQL Server is running, run mysql_setup.sql in Workbench, "
            "and verify the username/password. "
            f"MySQL reported: {error.msg}"
        ) from error

    if not connection.is_connected():
        connection.close()
        raise RuntimeError("MySQL connection did not become active.")
    return connection


def run_query(connection, query, params=(), changes_data=False, display=True):
    if not is_allowed_query(query):
        raise ValueError("Safety check rejected this SQL; only supported employee queries may run.")

    import mysql.connector

    cursor=connection.cursor()
    try:
        cursor.execute(query, params or None)
        if changes_data:
            connection.commit()
            if display:
                print(f"{cursor.rowcount} row(s) affected")
            return cursor.rowcount
        columns=cursor.column_names
        rows=cursor.fetchmany(100)
    except mysql.connector.Error as error:
        if changes_data:
            connection.rollback()
        raise RuntimeError(f"MySQL could not run the supported query: {error.msg}") from error
    finally:
        cursor.close()

    if display:
        print(" | ".join(columns))
        if rows:
            for row in rows:
                print(" | ".join(str(value) for value in row))
        else:
            print("(No matching rows.)")
        return rows
    return columns, rows


def answer_question(question, models, input_fn=input):
    ann, intent_vec, rf, ambiguity_vec=models
    intent=str(ann.predict(intent_vec.transform([question]))[0])
    rf.predict(ambiguity_vec.transform([question]))

    if not subject_is_supported(intent, question):
        raise ValueError("I could not match this request to a supported employee query.")

    kind, prompt=clarification_for(intent, question)
    if kind == "unsupported":
        raise ValueError(prompt)
    if kind:
        answer=input_fn(prompt).strip()
        if answer.lower() in {"quit", "exit", "q"}:
            return None
        if kind == "department":
            selected=next(
                (name for name in DEPARTMENTS if name.lower() == answer.lower()),
                None,
            )
            if not selected:
                raise ValueError("Please enter one of the listed department names.")
            question=f"{question} in {selected} department"
        elif kind == "department_update":
            selected=next(
                (name for name in DEPARTMENTS if name.lower() == answer.lower()),
                None,
            )
            if not selected:
                raise ValueError("Please enter one of the listed department names.")
            question=f"{question} to {selected}"
        elif kind == "delete":
            employee_id=employee_id_from(answer)
            if employee_id is None and answer.isdigit():
                employee_id=int(answer)
            if employee_id is None or employee_id < 1:
                raise ValueError("Please enter a positive employee ID.")
            question=f"{question} employee id {employee_id}"
        elif kind == "name_update":
            if re.search(r"\bto\b", answer):
                question=f"{question} {answer}"
            else:
                question=f"{question} to {answer}"
        elif kind == "ranking":
            if "salary" not in answer.lower():
                raise ValueError("This demo can rank employees only by salary.")
            question="Find the highest salary"
        else:
            question=f"{question} {answer}"

        intent=str(ann.predict(intent_vec.transform([question]))[0])
        rf.predict(ambiguity_vec.transform([question]))

    return make_query(intent, question)


def connect_mysql_for_pet(prompt_fn):
    try:
        import mysql.connector
    except ImportError as error:
        raise RuntimeError(
            "MySQL connector is missing. Install it with "
            "`python -m pip install mysql-connector-python`."
        ) from error

    host=os.environ.get("MYSQL_HOST", "127.0.0.1")
    port=int(os.environ.get("MYSQL_PORT", "3306"))
    user=os.environ.get("MYSQL_USER") or prompt_fn("MySQL username [root]: ").strip() or "root"
    password=os.environ.get("MYSQL_PASSWORD")
    if password is None:
        password=prompt_fn("MySQL password: ", secret=True)

    try:
        connection=mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="company_db",
            connection_timeout=5,
        )
    except mysql.connector.Error as error:
        raise RuntimeError(
            f"Could not connect to MySQL at {host}:{port} as {user}. "
            "Check that MySQL Server is running, run mysql_setup.sql in Workbench, "
            "and verify the username/password. "
            f"MySQL reported: {error.msg}"
        ) from error

    if not connection.is_connected():
        connection.close()
        raise RuntimeError("MySQL connection did not become active.")
    return connection


def run_pet_stdio():
    from url_classifier import is_url_input, predict_url

    def send_event(event):
        print(json.dumps(event, ensure_ascii=True, default=str), flush=True)

    def read_message():
        line=sys.stdin.readline()
        if not line:
            raise EOFError("The desktop pet closed the query connection.")
        message=json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("The desktop pet sent an invalid message.")
        return message

    def prompt_user(text, secret=False):
        send_event({
            "type": "prompt",
            "text": text,
            "secret": secret,
        })
        message=read_message()
        if message.get("type") != "reply" or not isinstance(message.get("text"), str):
            raise ValueError("Expected a text reply from the desktop pet.")
        return message["text"]

    connection=None
    models=None
    try:
        send_event({"type": "ready"})

        while True:
            message=read_message()
            if message.get("type") == "banana_image":
                try:
                    from banana_classifier import predict_banana

                    prediction=predict_banana(message.get("text", ""))
                    send_event({"type": "banana_result", **prediction})
                except (ImportError, OSError, RuntimeError, ValueError) as error:
                    send_event({"type": "error", "text": str(error)})
                continue
            if message.get("type") != "question":
                raise ValueError("Expected a question or banana image from the desktop pet.")
            question=message.get("text")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                send_event({"type": "error", "text": "Enter a question of 1 to 2000 characters."})
                continue

            try:
                question=question.strip()
                if is_url_input(question):
                    prediction=predict_url(question)
                    send_event({"type": "url_result", **prediction})
                    continue

                if models is None:
                    send_event({"type": "status", "text": "Training the query models..."})
                    models=train_models()
                    send_event({"type": "status", "text": "Connecting to the employee database..."})
                    connection=connect_mysql_for_pet(prompt_user)

                result=answer_question(question.strip(), models, input_fn=prompt_user)
                if result is None:
                    send_event({"type": "result", "cancelled": True})
                    continue
                query, params, changes_data=result
                if changes_data:
                    confirmation=prompt_user(
                        "This will change employee data. Type yes to continue:"
                    ).strip().lower()
                    if confirmation not in {"yes", "y"}:
                        send_event({"type": "result", "cancelled": True})
                        continue

                query_result=run_query(
                    connection,
                    query,
                    params,
                    changes_data,
                    display=False,
                )
                if changes_data:
                    send_event({
                        "type": "result",
                        "changesData": True,
                        "rowsAffected": query_result,
                    })
                else:
                    columns, rows=query_result
                    send_event({
                        "type": "result",
                        "columns": columns,
                        "rows": [list(row) for row in rows],
                    })
            except (ImportError, OSError, ValueError, RuntimeError) as error:
                send_event({"type": "error", "text": str(error)})
    except EOFError:
        pass
    except (json.JSONDecodeError, RuntimeError, ValueError) as error:
        send_event({"type": "error", "text": str(error)})
    finally:
        if connection is not None:
            connection.close()


def main():
    models=train_models()
    try:
        connection=connect_mysql()
    except RuntimeError as error:
        print(f"Cannot connect to MySQL: {error}")
        return

    try:
        while True:
            question=input("Ask about employees: ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                break
            if not question:
                continue
            try:
                result=answer_question(question, models)
                if result is None:
                    continue
                query, params, changes_data=result
                if changes_data:
                    confirmation=input(
                        "This will change employee data. Type yes to continue: "
                    ).strip().lower()
                    if confirmation not in {"yes", "y"}:
                        continue
                if query:
                    run_query(connection, query, params, changes_data)
            except ValueError as error:
                print(f"Cannot run this request: {error}")
            except RuntimeError as error:
                print(f"MySQL query error: {error}")
    finally:
        connection.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--pet-stdio"]:
        run_pet_stdio()
    else:
        main()
