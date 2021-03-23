
class TableProcessing:

    def __init__(self, table_name, cur, conn):
        self.table_name = table_name
        self.cur = cur
        self.conn = conn

    def create_table(self):
        query = (
            """
            create table {table} (
                site varchar(255),
                fraudulent int
            )
            """).format(table=self.table_name)

        try:
            self.cur.execute(query)
            self.conn.commit()
            print('Success: Created table {}'.format(self.table_name))
        except:
            self.cur.execute('rollback;')
            print("Error: Table name already exists")

    def drop_table(self):
        query = (
            """
            drop table {table};
            """
        ).format(table=self.table_name)

        try:
            self.cur.execute(query)
            self.conn.commit()
            print('Success: Dropped table {}'.format(self.table_name))
        except:
            self.cur.execute('rollback;')
            print("Error: Could not drop table")

    def populate_table_fromcsv(self, upload_path):
        query = (
            """
            copy {table}
            from '{file}' delimiter ',' csv;
            """).format(table=self.table_name, file=upload_path)

        try:
            self.cur.execute(query)
            self.conn.commit()
            print("Success: Records added to table")
        except:
            self.cur.execute('rollback;')
            print("Error: Inserting into table failed")
            pass

    def populate_table_manual(self, site, label):
        query = (
            """
            insert into {table}
            values ('{site}', {label})
            """).format(table=self.table_name, site=site, label=label)

        try:
            self.cur.execute(query)
            self.conn.commit()
            print("Success: Record added to table")
        except:
            self.cur.execute('rollback;')
            print("Error: Inserting into table failed")
            pass

    def delete_site_manual(self, site):
        query = (
            """
            delete from {table}
            where site='{site}'
            """).format(table=self.table_name, site=site)

        try:
            self.cur.execute(query)
            self.conn.commit()
            print("Success: Record deleted from table")
        except:
            self.cur.execute('rollback;')
            print("Error: Deleting record from table failed")
            pass

    def remove_duplicates(self):
        query = (
            """
            create table {table}_temp as
            select distinct * from {table};
            
            drop table {table};
            
            alter table {table}_temp rename to {table};
            """).format(table=self.table_name)

        try:
            self.cur.execute(query)
            self.conn.commit()
            print("Success: Duplicates removed")
        except:
            print("Error: Duplicates could not be removed")
            self.cur.execute('rollback;')
            pass


    def get_score(self, site):
        query = (
            """
            select fraudulent from {table} where site = '{site}';
            """
        ).format(table=self.table_name, site=site)

        try:
            self.cur.execute(query)
            score = self.cur.fetchall()[0][0]
            return score
        except:
            self.cur.execute('rollback;')
            pass
            return None
