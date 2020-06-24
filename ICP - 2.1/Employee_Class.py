class Employee:
    """
    Class for individual Employees
    """
    count = 0
    total = 0

    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        self.hours = "Part Time"
        Employee.count += 1
        Employee.total += salary


class FullTimeEmployee(Employee):
    """
    Inherits from parent class Employee but for full time
    """

    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)
        self.hours = "Full Time"


def avg_salary() -> float:
    """
    Returns average of all employees salaries
    """
    avg = Employee.total / Employee.count
    return round(avg, 2)


if __name__ == "__main__":

    # EXAMPLE INSTANCES FOR EMPLOYEE AND FULLTIMEEMPLOYEE
    dave = Employee("Dave", "Person", 30000, "electrical")
    kim = Employee("Kim", "Bottle", 35000, "mechanical")
    ben = FullTimeEmployee("Ben", "Ricardo", 67000, "civil")
    sally = FullTimeEmployee("Sally", "Stuff", 78000, "mechanical")

    # OUTPUT OF EACH EXAMPLE INSTANCE
    print(dave.name, dave.family, dave.salary, dave.department, dave.hours)
    print(kim.name, kim.family, kim.salary, kim.department, kim.hours)
    print(ben.name, ben.family, ben.salary, ben.department, ben.hours)
    print(sally.name, sally.family, sally.salary, sally.department, sally.hours)

    # OUTPUT OF NUMBER OF ALL EMPLOYEES AND AVG SALARIES
    print("The number of all employees is", Employee.count)
    print("The average salary of all employees is", avg_salary())
