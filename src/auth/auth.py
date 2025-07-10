import os
import boto3
import requests
from urllib.parse import parse_qs, urlparse
from lxml import html

def get_direct_access_token(username, password):
    """
    Get DESTINE access token directly using provided username and password.
    """
    SERVICE_URL = "http://localhost:5000"
    IAM_URL = "https://auth.destine.eu"
    IAM_REALM = "desp"
    IAM_CLIENT = "dcms_client"

    with requests.Session() as s:
        # Get the auth url
        response = s.get(
            url=f"{IAM_URL}/realms/{IAM_REALM}/protocol/openid-connect/auth",
            params={
                "client_id": IAM_CLIENT,
                "redirect_uri": SERVICE_URL,
                "scope": "openid",
                "response_type": "code",
            },
        )
        response.raise_for_status()
        auth_url = html.fromstring(response.content.decode()).forms[0].action

        # Login and get auth code
        login = s.post(
            auth_url,
            data={
                "username": username,
                "password": password,
            },
            allow_redirects=False,
        )

        if login.status_code == 200:
            tree = html.fromstring(login.content)
            error_message_element = tree.xpath('//span[@id="input-error"]/text()')
            error_message = (
                error_message_element[0].strip()
                if error_message_element
                else "Authentication failed"
            )
            print(f"Error: {error_message}")
            return None

        if login.status_code != 302:
            print(f"Login failed with status code: {login.status_code}")
            return None

        auth_code = parse_qs(urlparse(login.headers["Location"]).query)["code"][0]

        # Use the auth code to get the token
        response = requests.post(
            f"{IAM_URL}/realms/{IAM_REALM}/protocol/openid-connect/token",
            data={
                "client_id": IAM_CLIENT,
                "redirect_uri": SERVICE_URL,
                "code": auth_code,
                "grant_type": "authorization_code",
                "scope": "",
            },
        )

        if response.status_code != 200:
            print(f"Failed to get token. Status code: {response.status_code}")
            return None

        token_data = response.json()
        return {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token")
        }


class S3Connector:
    """A clean connector for S3-compatible storage services"""

    def __init__(self, endpoint_url, access_key_id,
                 secret_access_key, region_name='default'):
        """Initialize the S3Connector with connection parameters"""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

        # Create session
        self.session = boto3.session.Session()

        # Initialize S3 resource
        self.s3 = self.session.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

        # Initialize S3 client
        self.s3_client = self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def get_s3_client(self):
        """Get the boto3 S3 client"""
        return self.s3_client

    def get_s3_resource(self):
        """Get the boto3 S3 resource"""
        return self.s3

    def get_bucket(self, bucket_name):
        """Get a specific bucket by name"""
        return self.s3.Bucket(bucket_name)

    def list_buckets(self):
        """List all available buckets"""
        response = self.s3_client.list_buckets()
        if 'Buckets' in response:
            return [bucket['Name'] for bucket in response['Buckets']]
        return []

# if __name__ == "__main__":


#     from dotenv import load_dotenv
#     load_dotenv()
#     # Get credentials from environment variables
#     ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
#     SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
#     ENDPOINT_URL = 'https://eodata.dataspace.copernicus.eu'
#     # Initialize the connector
#     s3_connector = S3Connector(
#         endpoint_url=ENDPOINT_URL,
#         access_key_id=ACCESS_KEY_ID,
#         secret_access_key=SECRET_ACCESS_KEY
#     )
#     # Connect to S3
#     s3_connector.connect()
#     s3_client = s3_connector.get_s3_client()


