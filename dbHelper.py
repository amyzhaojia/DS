import adal, uuid, time,struct, pyodbc, os, logging
import pandas as pd
# # for public begin
# client_id = os.environ.get('Atom_DBAADClientId')
# client_secret = os.environ.get('Atom_DBAADClientSecret')
# tenant = os.environ.get('Atom_Tenant')
# constr= os.environ.get('Atom_ODBCConnectionString')
# # for public end

# for local test begin
client_id = '2b6be73a-6da6-42c4-a3e1-07f53533e880'
client_secret = 'NbpA4Xb2=uXUY[Q1plOb@qpNE[DiRpq8'
tenant = '72f988bf-86f1-41af-91ab-2d7cd011db47'
constr= 'Driver={ODBC Driver 17 for SQL Server};Server=tcp:atomstage.database.windows.net,1433;Database=atomstage1'
# for local test end

authorityHostUrl = "https://login.microsoftonline.com"
authority_url = authorityHostUrl + '/' + tenant
resource = "https://database.windows.net/"
context = adal.AuthenticationContext(authority_url, api_version=None)


token = context.acquire_token_with_client_credentials(
    resource,
    client_id,
    client_secret)

tokenb = bytes(token["accessToken"], 'utf-8')
exptoken = b''
for i in tokenb:
    exptoken += bytes({i})
    exptoken += bytes(1)
tokenstruct = struct.pack("=i", len(exptoken)) + exptoken


def get_connection():
    return pyodbc.connect(constr, attrs_before={1256: tokenstruct})


def execute_query(sql):
    result = None
    retry = 3
    while retry > 0:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            try:
                result = cursor.execute(sql).fetchall()
            except:
                conn.close()
                raise
            conn.close()
            retry = 0
        except Exception as ex:
            retry -= retry
            if retry == 0:
                logging.error("execute query Exception:" + str(ex))
                raise ex
            time.sleep(1)

    return result


def execute_query_df(sql):
    result = None
    retry = 3
    while retry > 0:
        try:
            conn = get_connection()
            try:
                result = pd.read_sql(sql, conn)
            except:
                conn.close()
                raise
            conn.close()
            retry = 0
        except Exception as ex:
            retry -= retry
            if retry == 0:
                logging.error("execute query Exception:" + str(ex))
                raise ex
            time.sleep(1)

    return result


def execute_batch_insert(pre_sql, sql, datas):
    '''
    Execute many sqls
    :return: 
    '''
    errorOccurred = True
    retry = 3
    while (retry > 0):
        try:
            conn = get_connection()
            try:
                cursor=conn.cursor()
                if pre_sql is not None:
                    cursor.execute(pre_sql)
                    conn.commit()
                cursor.fast_executemany = True
                cursor.executemany(sql, datas)
                conn.commit()
            except:
                conn.close()
                raise
            conn.close()
            retry = 0
            errorOccurred = False
        except Exception as ex:
            if retry==1:
                raise ex
            else:
                retry -= retry
                logging.error("execute many Exception:" + str(ex))
                time.sleep(1)
    return errorOccurred