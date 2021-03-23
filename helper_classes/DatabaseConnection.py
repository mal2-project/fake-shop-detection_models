import psycopg2
from configparser import ConfigParser

class DatabaseConnection:

    def config(self, filename='db.config', section='postgresql'):
        # create a parser
        parser = ConfigParser()
        # read config file
        parser.read(filename)

        # get section, default to postgresql
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))

        return db

    def connect(self):
        """ Connect to the PostgreSQL database server """
        conn = None
        try:
            # read connection parameters and connect
            params = self.config()
            conn = psycopg2.connect(**params)

            # create a cursor
            cur = conn.cursor()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        return cur, conn

    def disconnect(self, cur, conn):
        # close the communication with the PostgreSQL
        cur.close()
        conn.commit()

        if conn is not None:
            conn.close()
