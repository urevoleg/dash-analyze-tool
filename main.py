import json
import requests
import pandas as pd


class IotViewer:

    def __init__(self, kwargs):git
        self.base_url = kwargs['url']
        self.api_version = kwargs['version']
        self.model = kwargs['model']
        self.method = kwargs['method']
        self.url_ = '/'.join([url_part.strip('/') for url_part in
                              [self.base_url, 'api', f'v{self.api_version}', self.model, self.method]])
        self.method_params = kwargs['method_params']
        self.result = {'errors': None, 'data': None}
        self.response = None

    def get_serialized_(self, http_response):
        """
        Пока заглушка в виде try\except
        """
        try:
            self.result['data'] = json.loads(http_response.content)
        except Exception as e:
            self.result['errors'] = str(e)

    def make_request(self):
        req = requests.get(url=self.url_, params=self.method_params)
        self.response = req
        self.get_serialized_(req)
        return self.result

if __name__ == '__main__':
    p = dict(url='http://192.168.55.135:5000',
             version=1,
             model='weather',
             method='getgr',
             method_params=dict(period='hours', count=3, interval='minute', cols='pout'))
    iv = IotViewer(p)

    res = iv.make_request()

    df_api = pd.DataFrame(res['data'])\
        .assign(ts=lambda row: pd.to_datetime(row['ts']).dt.tz_convert('Europe/Moscow').dt.tz_localize(None))

    print(df_api.head())