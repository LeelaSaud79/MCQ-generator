import mysql.connector

# Connect to MySQL database
try:
    connection = mysql.connector.connect(
        host='localhost',
        database='test',  # Your database name
        user='root',      # Your MySQL username
        password=''       # Your MySQL password
    )

    if connection.is_connected():
        print("Connected to MySQL database")

        # Insert data into login table
        cursor = connection.cursor()
        insert_query = "INSERT INTO login (username, password) VALUES (%s, %s)"
        data = ('your_username', 'your_password')  # Replace with actual username and password
        cursor.execute(insert_query, data)
        connection.commit()
        print("Data inserted successfully")

except mysql.connector.Error as error:
    print("Error while connecting to MySQL", error)

finally:
    if 'connection' in locals():
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
