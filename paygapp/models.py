import pandas as pd
import os
from db_client import DB_Client

class File():

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = self.file_path.split('/')[-1].split('.')[0]
        self.db = DB_Client()

    def create_table(self):
        """reads file data and creates a table in database with filename"""
        file_df = pd.read_csv(self.file_path)
        columns = file_df.columns
        data_type = []
        for col in columns:
            if file_df[col].dtype == int:
                data_type.append((col, 'int(10)'))
            elif file_df[col].dtype == object:
                data_type.append((col, 'varchar(20)'))
        
        variables = ''.join(['{} {}, '.format(col, dtype) for col, dtype in data_type])[:-2]

        try:
            drop_query = """DROP TABLE {table_name}""".format(table_name=self.file_name)
            result = self.db.exec_query(drop_query)
        except:
            pass

        query = """CREATE TABLE {file_name} ({variables});
        """.format(file_name=self.file_name, variables=variables)
        print (query)

        # query to create file
        result = self.db.exec_query(query)


    def upload_to_db(self):
        """uploads a file to db"""

        query = """LOAD DATA LOCAL INFILE '{file_path}' INTO TABLE {table_name} FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\n' IGNORE 1 ROWS;
        """.format(file_path=self.file_path, table_name=self.file_name)

        result = self.db.exec_query(query)

        print (result)

    def read_records(self):
        """read all records from table"""
        query = """SELECT * FROM {table_name}""".format(table_name=self.file_name)


# ob = File("/home/shivam/Downloads/paygapp-master_version4/uploads/Glass_Door_data.csv")
# ob.create_table()
# ob.upload_to_db()