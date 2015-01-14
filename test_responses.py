from __future__ import (
    absolute_import, print_function, division, unicode_literals
)

import mock
import re
import requests
import responses
import pytest
try: 
    import grequests
except:
    grequests = None
    pass

from inspect import getargspec
from requests.exceptions import ConnectionError, HTTPError

xfail = pytest.mark.xfail()

def assert_reset():
    assert len(responses._default_mock._urls) == 0
    assert len(responses.calls) == 0

def assert_response(resp, body=None):
    assert resp.status_code == 200
    assert resp.headers['Content-Type'] == 'text/plain'
    assert resp.text == body


def test_response():
    @responses.activate
    def run():
        responses.add(responses.GET, 'http://example.com', body=b'test')
        resp = requests.get('http://example.com')
        assert_response(resp, 'test')
        assert len(responses.calls) == 1
        assert responses.calls[0].request.url == 'http://example.com/'
        assert responses.calls[0].response.content == b'test'

        resp = requests.get('http://example.com?foo=bar')
        assert_response(resp, 'test')
        assert len(responses.calls) == 2
        assert responses.calls[1].request.url == 'http://example.com/?foo=bar'
        assert responses.calls[1].response.content == b'test'

    run()
    assert_reset()


def test_connection_error():
    @responses.activate
    def run():
        responses.add(responses.GET, 'http://example.com')

        with pytest.raises(ConnectionError):
            requests.get('http://example.com/foo')

        assert len(responses.calls) == 1
        assert responses.calls[0].request.url == 'http://example.com/foo'
        assert type(responses.calls[0].response) is ConnectionError

    run()
    assert_reset()


def test_match_querystring():
    @responses.activate
    def run():
        url = 'http://example.com?test=1&foo=bar'
        responses.add(
            responses.GET, url,
            match_querystring=True, body=b'test')
        resp = requests.get('http://example.com?test=1&foo=bar')
        assert_response(resp, 'test')
        resp = requests.get('http://example.com?foo=bar&test=1')
        assert_response(resp, 'test')

    run()
    assert_reset()


def test_match_querystring_error():
    @responses.activate
    def run():
        responses.add(
            responses.GET, 'http://example.com/?test=1',
            match_querystring=True)

        with pytest.raises(ConnectionError):
            requests.get('http://example.com/foo/?test=2')

    run()
    assert_reset()


def test_match_querystring_regex():
    @responses.activate
    def run():
        """Note that `match_querystring` value shouldn't matter when passing a
        regular expression"""

        responses.add(
            responses.GET, re.compile(r'http://example\.com/foo/\?test=1'),
            body='test1', match_querystring=True)

        resp = requests.get('http://example.com/foo/?test=1')
        assert_response(resp, 'test1')

        responses.add(
            responses.GET, re.compile(r'http://example\.com/foo/\?test=2'),
            body='test2', match_querystring=False)

        resp = requests.get('http://example.com/foo/?test=2')
        assert_response(resp, 'test2')

    run()
    assert_reset()


def test_match_querystring_error_regex():
    @responses.activate
    def run():
        """Note that `match_querystring` value shouldn't matter when passing a
        regular expression"""

        responses.add(
            responses.GET, re.compile(r'http://example\.com/foo/\?test=1'),
            match_querystring=True)

        with pytest.raises(ConnectionError):
            requests.get('http://example.com/foo/?test=3')

        responses.add(
            responses.GET, re.compile(r'http://example\.com/foo/\?test=2'),
            match_querystring=False)

        with pytest.raises(ConnectionError):
            requests.get('http://example.com/foo/?test=4')

    run()
    assert_reset()


def test_accept_string_body():
    @responses.activate
    def run():
        url = 'http://example.com/'
        responses.add(
            responses.GET, url, body='test')
        resp = requests.get(url)
        assert_response(resp, 'test')

    run()
    assert_reset()


def test_throw_connection_error_explicit():
    @responses.activate
    def run():
        url = 'http://example.com'
        exception = HTTPError('HTTP Error')
        responses.add(
            responses.GET, url, exception)

        with pytest.raises(HTTPError) as HE:
            requests.get(url)

        assert str(HE.value) == 'HTTP Error'

    run()
    assert_reset()


def test_callback():
    body = b'test callback'
    status = 400
    headers = {'foo': 'bar'}
    url = 'http://example.com/'

    def request_callback(request):
        return (status, headers, body)

    @responses.activate
    def run():
        responses.add_callback(responses.GET, url, request_callback)
        resp = requests.get(url)
        assert resp.text == "test callback"
        assert resp.status_code == status
        assert 'foo' in resp.headers
        assert resp.headers['foo'] == 'bar'

    run()
    assert_reset()


def test_regular_expression_url():
    @responses.activate
    def run():
        url = re.compile(r'https?://(.*\.)?example.com')
        responses.add(responses.GET, url, body=b'test')

        resp = requests.get('http://example.com')
        assert_response(resp, 'test')

        resp = requests.get('https://example.com')
        assert_response(resp, 'test')

        resp = requests.get('https://uk.example.com')
        assert_response(resp, 'test')

        with pytest.raises(ConnectionError):
            requests.get('https://uk.exaaample.com')

    run()
    assert_reset()

@xfail(reason="requests before v1.0  does not support adapters")
def test_custom_adapter():
    @responses.activate
    def run():
        url = "http://example.com"
        responses.add(responses.GET, url, body=b'test')

        class DummyAdapter(requests.adapters.HTTPAdapter):
            pass

        # Test that the adapter is actually used
        adapter = mock.Mock(spec=DummyAdapter())
        session = requests.Session()
        session.mount("http://", adapter)

        resp = session.get(url)
        assert adapter.build_response.called == 1

        # Test that the response is still correctly emulated
        session = requests.Session()
        session.mount("http://", DummyAdapter())

        resp = session.get(url)
        assert_response(resp, 'test')

    run()


def test_responses_as_context_manager():
    def run():
        with responses.mock:
            responses.add(responses.GET, 'http://example.com', body=b'test')
            resp = requests.get('http://example.com')
            assert_response(resp, 'test')
            assert len(responses.calls) == 1
            assert responses.calls[0].request.url == 'http://example.com/'
            assert responses.calls[0].response.content == b'test'

            resp = requests.get('http://example.com?foo=bar')
            assert_response(resp, 'test')
            assert len(responses.calls) == 2
            assert (responses.calls[1].request.url ==
                    'http://example.com/?foo=bar')
            assert responses.calls[1].response.content == b'test'

    run()
    assert_reset()


def test_activate_doesnt_change_signature():
    def test_function(a, b=None):
        return (a, b)

    decorated_test_function = responses.activate(test_function)
    assert getargspec(test_function) == getargspec(decorated_test_function)
    assert decorated_test_function(1, 2) == test_function(1, 2)
    assert decorated_test_function(3) == test_function(3)


def test_activate_doesnt_change_signature_for_method():
    class TestCase(object):

        def test_function(self, a, b=None):
            return (self, a, b)

    test_case = TestCase()
    argspec = getargspec(test_case.test_function)
    decorated_test_function = responses.activate(test_case.test_function)
    assert argspec == getargspec(decorated_test_function)
    assert decorated_test_function(1, 2) == test_case.test_function(1, 2)
    assert decorated_test_function(3) == test_case.test_function(3)


@pytest.mark.skipif(grequests is None,
    reason="requires grequests")
def test_response_grequests():
    @responses.activate
    def run():
        responses.add(responses.GET, 'http://example.com', body=b'test')
        resp = grequests.map((grequests.get('http://example.com'),))[0]
        assert_response(resp, 'test')
        assert len(responses.calls) == 1
        assert responses.calls[0].request.url == 'http://example.com/'
        assert responses.calls[0].response.content == b'test'

        resp = requests.get('http://example.com?foo=bar')
        assert_response(resp, 'test')
        assert len(responses.calls) == 2
        assert responses.calls[1].request.url == 'http://example.com/?foo=bar'
        assert responses.calls[1].response.content == b'test'

    run()
    assert_reset()


def test_allow_redirects():
    redirecting_url = 'http://example.com'
    final_url = 'http://example.com/2'
    url_re = re.compile(r'^http://example.com(/)?(\d+)?$')

    def request_callback(request):
        if request.url.endswith('/2'):
            return 200, (), b'test'
        else:
            if request.url.endswith('0'):
                n = 1
            elif request.url.endswith('1'):
                n = 2
            else:
                n = 0
            redirect_headers = {'location': '/{!s}'.format(n)}
            return 301, redirect_headers, None

    @responses.activate
    def run():
        # setup redirect
        responses.add_callback(responses.GET, url_re, request_callback)

        resp_no_redirects = requests.get(redirecting_url, allow_redirects=False)
        assert resp_no_redirects.status_code == 301
        assert resp_no_redirects.url != final_url

        resp_yes_redirects = requests.get(redirecting_url, allow_redirects=True)
        assert len(resp_yes_redirects.history) == 3
        assert resp_yes_redirects.status_code == 200
        assert resp_yes_redirects.url == final_url

    run()
    assert_reset()