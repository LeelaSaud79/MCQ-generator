import mysql.connector

class UserDatabase:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None

    def connect(self):
        self.conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def create_table(self):
        self.connect()
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INT AUTO_INCREMENT PRIMARY KEY, 
                     username VARCHAR(255) UNIQUE, 
                     email VARCHAR(255) UNIQUE, 
                     password VARCHAR(255))''')
        self.conn.commit()
        self.disconnect()

    def insert_user(self, username, email, password):
        self.connect()
        c = self.conn.cursor()
        c.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
        self.conn.commit()
        self.disconnect()

    def check_user(self, email, password):
        self.connect()
        c = self.conn.cursor()
        c.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = c.fetchall()
        self.disconnect()
        return user
