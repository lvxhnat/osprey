from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import pandas_gbq
import pandas as pd
import gcsfs

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/lohyikuang/Documents/Python/Big Query/Credentials/bigquery_credentials.json'
credentials = service_account.Credentials.from_service_account_file('/Users/lohyikuang/Documents/Python/Big Query/Credentials/bigquery_credentials.json')
project_id = 'root-sanctuary-178203'
client = bigquery.Client(credentials= credentials,project=project_id)
# Update the in-memory credentials cache (added in pandas-gbq 0.7.0).
pandas_gbq.context.credentials = credentials
pandas_gbq.context.project = project_id
# fs = gcsfs.GCSFileSystem(
#     project='root-sanctuary-178203',
#     token= '/Users/synthesis/Documents/Python/Big Query/Credentials/cloudstorage_credentials.json'
# )
fs = gcsfs.GCSFileSystem(
    project='root-sanctuary-178203',
    token= os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

def get_all_files(bucket_name = "synthesisbucket"):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)
def get_files_with_prefix(bucket_name = "synthesisbucket", prefix = "", delimiter=None):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    blob_names = []
    for blob in blobs:
        blob_names.append(blob.name)
    if delimiter:
        print('Prefixes:')
        for prefix in blobs.prefixes:
            print(prefix)
    return blob_names

def read_file_simple(filepath):
    try:
        with fs.open(filepath) as f: 
            df = pd.read_csv(f, sep="\t", encoding = "utf8")
        return df 
    except:
        print(filepath)
        return None
    
def read_files_from_gcs(folder, bucket_name = "synthesisbucket", prefix = "", delimiter=None):
    read_base = "gs://" + bucket_name + "/"
    blobs = get_files_with_prefix(bucket_name, prefix = folder)
    all_data = pd.DataFrame()
    temp = pd.DataFrame()
    files_done = 0
    for blob in blobs:
        with fs.open(read_base + blob) as f:
            df = pd.read_csv(f, sep="\t", encoding = "utf8")
            temp = pd.concat([temp, df])
            if(files_done % 1000 == 0):
                all_data = pd.concat([all_data, temp])
                files_done = 0
                temp = pd.DataFrame()
            files_done = files_done + 1
    all_data = pd.concat([all_data, temp])
    return all_data

def read_files_gcs(file_path):
    path_to_gcs_folder = file_path

    gcs_files = get_files_with_prefix('synthesisbucket', path_to_gcs_folder)
    gcs_files = ['gs://synthesisbucket/'+file for file in gcs_files]

    p = Pool(8)

    all_dfs = p.map(read_file_simple, gcs_files)
    p.close()
    df = pd.concat(all_dfs, sort=False)
    df = df.drop_duplicates()
    return df

def queryBigQueryWithQuery(query):
    return pandas_gbq.read_gbq(query)      

def queryBigQueryGeneric(database, tablename, columns="*"):
    if(type(columns) == "list"):
        columns = ",".join(columns)
    query = "SELECT %s FROM `%s.%s`" % (columns, database, tablename)
    return queryBigQueryWithQuery(query)

def writeToBigQuery(df, database, tablename, mode="append"):
    destination = "%s.%s" % (database, tablename)
    df.to_gbq(destination_table = destination, 
              if_exists=mode)
    
def writeToCloudStorage(df, write_path, bucket="synthesisbucket"):
    with fs.open("gs://" + bucket + "/" + write_path, "w") as f:
        df.to_csv(f, encoding="utf8", index=None, sep="\t")