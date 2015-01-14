"""
Copyright 2013 Dropbox, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import (
    absolute_import, print_function, division, unicode_literals
)

import re
import six

if six.PY2:
    try:
        from six import cStringIO as BufferIO
    except ImportError:
        from six import StringIO as BufferIO
else:
    from io import BytesIO as BufferIO

import inspect
from collections import namedtuple, Sequence, Sized
from functools import update_wrapper
from requests.exceptions import ConnectionError, TooManyRedirects
from requests.models import Response, Request
from requests.structures import CaseInsensitiveDict
from requests.utils import get_encoding_from_headers, requote_uri
from requests.status_codes import codes

try:
    from requests.packages.urllib3.response import HTTPResponse
except ImportError:
    from urllib3.response import HTTPResponse

if six.PY2:
    from urlparse import urlparse, parse_qsl
else:
    from urllib.parse import urlparse, parse_qsl

from requests.compat import (
    cookielib, urljoin
)


Call = namedtuple('Call', ['request', 'response'])

_wrapper_template = """\
def wrapper%(signature)s:
    with responses:
        return func%(funcargs)s
"""

REDIRECT_STATI = (codes.moved, codes.found, codes.other, codes.temporary_moved)


# mostly pulled from requests.models.Request._build_response
def resolve_redirects(response, request, **resolve_kwargs):
    history = []

    if not response.status_code in REDIRECT_STATI:
        return

    while ('location' in response.headers):
        response.content  # Consume socket so it can be released

        if not len(history) < request.config.get('max_redirects'):
            raise TooManyRedirects()

        # Release the connection back into the pool.
        response.raw.release_conn()

        history.append(response)

        url = response.headers['location']
        data = None  # self.data
        files = None  # self.files

        # Handle redirection without scheme (see: RFC 1808 Section 4)
        if url.startswith('//'):
            parsed_rurl = urlparse(response.url)
            url = '%s:%s' % (parsed_rurl.scheme, url)

        # Facilitate non-RFC2616-compliant 'location' headers
        # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
        if not urlparse(url).netloc:
            url = urljoin(response.url,
                          # Compliant with RFC3986, we percent
                          # encode the url.
                          requote_uri(url))

        # http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3.4
        if response.status_code is codes.see_other:
            method = 'GET'
            data = None
            files = None
        else:
            method = request.method

        # Do what the browsers do if strict_mode is off...
        if (not request.config.get('strict_mode')):

            if response.status_code in (codes.moved, codes.found) and request.method == 'POST':
                method = 'GET'
                data = None
                files = None

            if (response.status_code == 303) and request.method != 'HEAD':
                method = 'GET'
                data = None
                files = None

        # Remove the cookie headers that were sent.
        headers = request.headers
        try:
            del headers['Cookie']
        except KeyError:
            pass

        request_next = Request(
            url=url,
            headers=headers,
            files=files,
            method=method,
            redirect=True,
            data=data,
            params=request.session.params,
            auth=request.auth,
            cookies=request.cookies,
            config=request.config,
            timeout=request.timeout,
            _poolmanager=request._poolmanager,
            proxies=request.proxies,
            verify=request.verify,
            session=request.session,
            cert=request.cert,
            prefetch=request.prefetch,
        )

        request_next.send()
        response = request_next.response
        response.history = history
        yield response

# mostly pulled from requests.models.Request._build_response
def build_response(req, resp):
    """Builds a :class:`Response <requests.Response>` object from a urllib3
    response. This should not be called from user code, and is only exposed
    for use when subclassing the
    :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`
    :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
    :param resp: The urllib3 response object.
    """
    response = Response()
    response.cookies = cookielib.CookieJar()

    # Fallback to None if there's no status_code, for whatever reason.
    response.status_code = getattr(resp, 'status', None)

    # Make headers case-insensitive.
    response.headers = CaseInsensitiveDict(getattr(resp, 'headers', {}))

    # Set encoding.
    response.encoding = get_encoding_from_headers(response.headers)
    response.raw = resp
    #already a property pointing to raw.reason in v0.14.2
    #response.reason = response.raw.reason

    if isinstance(req.url, bytes):
        response.url = req.url.decode('utf-8')
    else:
        response.url = req.url

    # Add new cookies from the server.
    #extract_cookies_to_jar(response.cookies, req, resp)

    # Give the Response some context.
    response.request = req
    response.connection = None

    return response


def get_wrapped(func, wrapper_template, evaldict):
    # Preserve the argspec for the wrapped function so that testing
    # tools such as pytest can continue to use their fixture injection.
    args, a, kw, defaults = inspect.getargspec(func)
    values = args[-len(defaults):] if defaults else None

    signature = inspect.formatargspec(args, a, kw, defaults)
    is_bound_method = hasattr(func, '__self__')
    if is_bound_method:
        args = args[1:]     # Omit 'self'
    callargs = inspect.formatargspec(args, a, kw, values,
                                     formatvalue=lambda v: '=' + v)

    ctx = {'signature': signature, 'funcargs': callargs}
    six.exec_(wrapper_template % ctx, evaldict)

    wrapper = evaldict['wrapper']

    update_wrapper(wrapper, func)
    if is_bound_method:
        wrapper = wrapper.__get__(func.__self__, type(func.__self__))
    return wrapper


class CallList(Sequence, Sized):
    def __init__(self):
        self._calls = []

    def __iter__(self):
        return iter(self._calls)

    def __len__(self):
        return len(self._calls)

    def __getitem__(self, idx):
        return self._calls[idx]

    def add(self, request, response):
        self._calls.append(Call(request, response))

    def reset(self):
        self._calls = []


def _is_string(s):
    return isinstance(s, (six.string_types, six.text_type))


def _ensure_url_default_path(url, match_querystring=False):
    # ensure the url has a default path set if the url is a string
    if _is_string(url) and url.count('/') == 2:
        if match_querystring:
            return url.replace('?', '/?', 1)  
        else:
            return url + '/'
    return url


class RequestsMock(object):
    DELETE = 'DELETE'
    GET = 'GET'
    HEAD = 'HEAD'
    OPTIONS = 'OPTIONS'
    PATCH = 'PATCH'
    POST = 'POST'
    PUT = 'PUT'

    def __init__(self):
        self._calls = CallList()
        self.reset()

    def reset(self):
        self._urls = []
        self._calls.reset()

    def add(self, method, url, body='', match_querystring=False,
            status=200, adding_headers=None, stream=False,
            content_type='text/plain'):

        url = _ensure_url_default_path(url, match_querystring)

        # body must be bytes
        if isinstance(body, six.text_type):
            body = body.encode('utf-8')

        self._urls.append({
            'url': url,
            'method': method,
            'body': body,
            'content_type': content_type,
            'match_querystring': match_querystring,
            'status': status,
            'adding_headers': adding_headers,
            'stream': stream,
        })

    def add_callback(self, method, url, callback, match_querystring=False,
                     content_type='text/plain'):
        # ensure the url has a default path set if the url is a string
        url = _ensure_url_default_path(url, match_querystring)

        self._urls.append({
            'url': url,
            'method': method,
            'callback': callback,
            'content_type': content_type,
            'match_querystring': match_querystring,
        })

    @property
    def calls(self):
        return self._calls

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()
        self.reset()

    def activate(self, func):
        evaldict = {'responses': self, 'func': func}
        return get_wrapped(func, _wrapper_template, evaldict)

    def _find_match(self, request):
        for match in self._urls:
            if request.method != match['method']:
                continue

            if not self._has_url_match(match, request.url):
                continue

            return match

        return None

    def _has_url_match(self, match, request_url):
        #url = _ensure_url_default_path(match['url'])
        url = match['url']

        if _is_string(url):
            if match['match_querystring']:
                return self._has_strict_url_match(url, request_url)
            else:
                #url_without_qs = _ensure_url_default_path(request_url.split('?', 1)[0])
                url_without_qs = request_url.split('?', 1)[0]
                return url == url_without_qs
        elif isinstance(url, re._pattern_type) and url.match(request_url):
            return True
        else:
            return False

    @staticmethod
    def _has_strict_url_match(url, other):
        url_parsed = urlparse(url)
        other_parsed = urlparse(other)

        if url_parsed[:3] != other_parsed[:3]:
            return False

        url_qsl = sorted(parse_qsl(url_parsed.query))
        other_qsl = sorted(parse_qsl(other_parsed.query))
        return url_qsl == other_qsl

    def _on_request(self, session, request, **kwargs):
        match = self._find_match(request)
        # TODO(dcramer): find the correct class for this
        if match is None:
            error_msg = 'Connection refused: {0}'.format(request.url)
            response = ConnectionError(error_msg)

            self._calls.add(request, response)
            raise response
        if 'body' in match and isinstance(match['body'], Exception):
            self._calls.add(request, match['body'])
            raise match['body']

        headers = {
            'Content-Type': match['content_type'],
        }
        if 'callback' in match:  # use callback
            status, r_headers, body = match['callback'](request)
            if isinstance(body, six.text_type):
                body = body.encode('utf-8')
            body = BufferIO(body)
            headers.update(r_headers)

        elif 'body' in match:
            if match['adding_headers']:
                headers.update(match['adding_headers'])
            status = match['status']
            body = BufferIO(match['body'])

        response = HTTPResponse(
            status=status,
            body=body,
            headers=headers,
            preload_content=False,
            original_response=namedtuple('OR', 'msg')(''),
        )
        response = build_response(request, response)
        response._content = body.getvalue()
        if not match.get('stream'):
            response.content  # NOQA

        self._calls.add(request, response)

        ## begin stuff for allow_redirects=True 
        if kwargs.get('allow_redirects', False):
            # include some redirect resolving logic from 
            # requests.sessions.Session
            resolve_kwargs = {k: v for (k, v) in kwargs.items() if
                              k in ('stream', 'timeout', 'cert', 'proxies')}
            
            # this recurses if response.is_redirect, 
            # but limited by session.max_redirects
            gen = resolve_redirects(response, request, **resolve_kwargs)

            history = [resp for resp in gen] if kwargs.get('allow_redirects') else []

            # Shuffle things around if there's history.
            if history:
                # Insert the first (original) request at the start
                history.insert(0, response)
                # Get the last request made
                response = history.pop()
                response.history = history
        #</allow_redirects>

        request.response = response
        return response

    def start(self):
        import mock

        def unbound_on_send(rself, *a, **kwargs):
            session = None
            rself.url = _ensure_url_default_path(rself.url, '?' in rself.url)
            kwargs.update({'allow_redirects': rself.allow_redirects})
            return self._on_request(session, rself, *a, **kwargs)
        self._patcher = mock.patch('requests.models.Request.send', unbound_on_send)
        self._patcher.start()

    def stop(self):
        self._patcher.stop()


# expose default mock namespace
mock = _default_mock = RequestsMock()
__all__ = []
for __attr in (a for a in dir(_default_mock) if not a.startswith('_')):
    __all__.append(__attr)
    globals()[__attr] = getattr(_default_mock, __attr)