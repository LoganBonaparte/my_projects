import sqlite3
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import heapq

class TaskManager:
    def __init__(self, db_name="tasks.db"):
        self.db_name = db_name
        self.create_table()

    # establish database connection
    def connect(self):
        return sqlite3.connect(self.db_name)

    # creates the task table if it doesn't exists
    def create_table(self):
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TAbLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        description TEXT,
                        priority INTEGER NOT NULL,
                        due_date TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'Pending'  
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    # adds a task
    def add_task(self):
        name = input ("Task name: ").strip()
        description = input("Description: ").strip()

        while True:
            try:
                priority = int(input("Priority (1-5, 1 = High, 5 = Low): "))
                if 1 <= priority <= 5:
                    break
                else:
                    print("Priority must be between 1 and 5")
            except ValueError:
                print("Invalid input. Enter a number between 1 and 5")

        while True:
            due_date = input("Due date (YYYY-MM-DD): ").strip()
            try:
                due_date_obj = datetime.datetime.strptime(due_date, "%Y-%m-%d").date()
                if due_date_obj >= datetime.date.today():
                    print(f"Your selected due date is: {due_date}")
                    break
                else:
                    print("Due date cannot be a past date")
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD")

        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO tasks (name, description, priority, due_date)
                    VALUES (?, ?, ?, ?)""",
                    (name, description, priority, due_date))
                conn.commit()
            print("Task added successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    # display all tasks sorting by priority by default. Due_date sorting is an option too
    def view_tasks(self, sort_by="priority"):
        try:
            with self.connect() as conn:
                df=pd.read_sql_query("SELECT * FROM tasks", conn)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return

        if df.empty:
            print("No tasks found")
            return

        if sort_by == "priority":
            df = df.sort_values(by="priority", ascending=True)
        elif sort_by == "due_date":
            df = df.sort_values(by="due_date")

        print("Task List: ")
        print(df.to_string(index=False))

    # Update an existing task
    def update_task(self):
        self.view_tasks()
        task_id = input("Enter task ID to update: ").strip()

        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                task = cursor.fetchone()

                if not task:
                    print("Task not found")
                    return

                print("\n1. Update description")
                print("2. Mark as completed")
                print("3. Cancel")

                while True:
                    choice = input("Choose an option: ").strip()

                    if choice == "1":
                        new_descr = input("Enter new description: ").strip()
                        cursor.execute("UPDATE tasks SET description = ? WHERE id = ?", (new_descr, task_id))
                        print("Description Updated")
                        break
                    elif choice == "2":
                        cursor.execute("UPDATE tasks SET status = 'Completed' WHERE id = ?", (task_id,))
                        print("Task marked as completed")
                        break
                    elif choice == "3":
                        print("Update cancelled")
                        return
                    else:
                        print("Please select a valid option")

                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    # Delets a task
    def delete_task(self):
        self.view_tasks()
        task_id = input("Enter task ID to delete: ").strip()

        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                task = cursor.fetchone()

                if not task:
                    print(f"Task {task_id} not found")
                    return

                cursor.execute("DELETE FROM tasks WHERE id = ?",(task_id,))
                conn.commit()
            print("Task deleted successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    # Suggest tasks using a priority queue
    def suggest_tasks(self):
        try:
            with self.connect() as conn:
                df = pd.read_sql_query("SELECT * FROM tasks WHERE status = 'Pending'", conn)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return

        if df.empty:
            print("No pending tasks")
            return

        tasks_heap = []
        for _, row in df.iterrows():
            heapq.heappush(tasks_heap, (row["priority"], row["due_date"], row["name"]))

        print("Suggested Task Order:")
        while tasks_heap:
            priority, due_date, name = heapq.heappop(tasks_heap)
            print(f"{name} (Priority: {priority}, Due: {due_date})")

    # Generate a gantt chart of tasks
    def generate_gantt(self):
        try:
            with self.connect() as conn:
                df = pd.read_sql_query("SELECT name, due_date FROM tasks WHERE status = 'Pending'", conn)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return

        if df.empty:
            print("No pending tasks to display")
            return

        df["due_date"] = pd.to_datetime(df["due_date"])
        df = df.sort_values(by="due_date")

        plt.figure(figsize=(10, 5))
        plt.barh(df["name"], df["due_date"].astype(str), color="skyblue")
        plt.xlabel("Due Date")
        plt.ylabel("Task")
        plt.title("Task Due Dates")
        plt.xticks(rotation=45)
        plt.show()

    # Export tasks to csv
    def export_csv(self, filename="exported_tasks.csv"):
        try:
            with self.connect() as conn:
                df = pd.read_sql_query("SELECT * FROM tasks", conn)
        except sqlite3.Error as e:
            print(f"Database Error: {e}")
            return

        if df.empty:
            print("No tasks to export")
            return

        df.to_csv(filename, index=False)
        print(f"Tasks successfully exported to '{filename}'")

def main():
        manager = TaskManager()

        while True:
            print("\n Task Manager Menu")
            print("1. Add Task")
            print("2. View Tasks")
            print("3. Update Task")
            print("4. Delete Task")
            print("5. Suggest Tasks")
            print("6. Generate Gantt Chart")
            print("7. Export tasks to CSV")
            print("8. Exit")

            choice = input("Choose an option: ").strip()

            if choice == "1":
                manager.add_task()
            elif choice == "2":
                sort = input("Sort by (priority / due_date): ").strip().lower()
                if sort not in ["priority", "due_date"]:
                    print("Invalid sort option. Default priority sort will be used.")
                    sort = "priority"
                manager.view_tasks(sort)
            elif choice == "3":
                manager.update_task()
            elif choice == "4":
                manager.delete_task()
            elif choice == "5":
                manager.suggest_tasks()
            elif choice == "6":
                manager.generate_gantt()
            elif choice == "7":
                manager.export_csv()
            elif choice == "8":
                print("Exiting Task Manager")
                break
            else:
                print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
