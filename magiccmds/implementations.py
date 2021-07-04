import timeit, time, gc, itertools, shlex, getopt, ast, traceback, sys, math, pstats, re, os, io, tempfile, ctypes, locale, warnings, re
import cProfile as profile
from pathlib import Path
from ctypes import c_int, POINTER
from ctypes.wintypes import LPCWSTR, HLOCAL
from inspect import stack
from shutil import get_terminal_size as _get_terminal_size
StringIO = io.StringIO

class CoreError(Exception):
    pass

class UsageError(CoreError):
    pass

esc_re = re.compile(r"(\x1b[^m]+m)")
#Below check completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/page.py#L324-L345
if os.name == 'nt' and os.environ.get('TERM','dumb') != 'emacs':
    import msvcrt
    def page_more():
        """ Smart pausing between pages
        @return:    True if need print more lines, False if quit
        """
        sys.stdout.write('---Return to continue, q to quit--- ')
        ans = msvcrt.getwch()
        if ans in ("q", "Q"):
            result = False
        else:
            result = True
        sys.stdout.write("\b"*37 + " "*37 + "\b"*37)
        return result
else:
    def page_more():
        ans = input('---Return to continue, q to quit--- ')
        if ans.lower().startswith('q'):
            return False
        else:
            return True

#Below class and next two assignments completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/inputtransformer2.py#L50-L114
class PromptStripper:
    """Remove matching input prompts from a block of input.
    Parameters
    ----------
    prompt_re : regular expression
        A regular expression matching any input prompt (including continuation,
        e.g. ``...``)
    initial_re : regular expression, optional
        A regular expression matching only the initial prompt, but not continuation.
        If no initial expression is given, prompt_re will be used everywhere.
        Used mainly for plain Python prompts (``>>>``), where the continuation prompt
        ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.
    Notes
    -----
    If initial_re and prompt_re differ,
    only initial_re will be tested against the first line.
    If any prompt is found on the first two lines,
    prompts will be stripped from the rest of the block.
    """
    def __init__(self, prompt_re, initial_re=None):
        self.prompt_re = prompt_re
        self.initial_re = initial_re or prompt_re
    
    def _strip(self, lines):
        return [self.prompt_re.sub('', l, count=1) for l in lines]
    
    def __call__(self, lines):
        if not lines:
            return lines
        if self.initial_re.match(lines[0]) or \
                (len(lines) > 1 and self.prompt_re.match(lines[1])):
            return self._strip(lines)
        return lines

classic_prompt = PromptStripper(
    prompt_re=re.compile(r'^(>>>|\.\.\.)( |$)'),
    initial_re=re.compile(r'^>>>( |$)')
)

ipython_prompt = PromptStripper(
    re.compile(
        r"""
        ^(                         # Match from the beginning of a line, either:
                                   # 1. First-line prompt:
        ((\[nav\]|\[ins\])?\ )?    # Vi editing mode prompt, if it's there
        In\                        # The 'In' of the prompt, with a space
        \[\d+\]:                   # Command index, as displayed in the prompt
        \                          # With a mandatory trailing space
        |                          # ... or ...
                                   # 2. The three dots of the multiline prompt
        \s*                        # All leading whitespace characters
        \.{3,}:                    # The three (or more) dots
        \ ?                        # With an optional trailing space
        )
        """,
        re.VERBOSE,
    )
)

#Below two functions and assigning of DEFAULT_ENCODING variable completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/encoding.py#L21-L71
def get_stream_enc(stream, default=None):
    """Return the given stream's encoding or a default.
    There are cases where ``sys.std*`` might not actually be a stream, so
    check for the encoding attribute prior to returning it, and return
    a default if it doesn't exist or evaluates as False. ``default``
    is None if not provided.
    """
    if not hasattr(stream, 'encoding') or not stream.encoding:
        return default
    else:
        return stream.encoding

# Less conservative replacement for sys.getdefaultencoding, that will try
# to match the environment.
# Defined here as central function, so if we find better choices, we
# won't need to make changes all over IPython.
def getdefaultencoding(prefer_stream=True):
    """Return IPython's guess for the default encoding for bytes as text.
    If prefer_stream is True (default), asks for stdin.encoding first,
    to match the calling Terminal, but that is often None for subprocesses.
    Then fall back on locale.getpreferredencoding(),
    which should be a sensible platform default (that respects LANG environment),
    and finally to sys.getdefaultencoding() which is the most conservative option,
    and usually UTF8 as of Python 3.
    """
    enc = None
    if prefer_stream:
        enc = get_stream_enc(sys.stdin)
    if not enc or enc=='ascii':
        try:
            # There are reports of getpreferredencoding raising errors
            # in some cases, which may well be fixed, but let's be conservative here.
            enc = locale.getpreferredencoding()
        except Exception:
            pass
    enc = enc or sys.getdefaultencoding()
    # On windows `cp0` can be returned to indicate that there is no code page.
    # Since cp0 is an invalid encoding return instead cp1252 which is the
    # Western European default.
    if enc == 'cp0':
        warnings.warn(
            "Invalid code page cp0 detected - using cp1252 instead."
            "If cp1252 is incorrect please ensure a valid code page "
            "is defined for the process.", RuntimeWarning)
        return 'cp1252'
    return enc

DEFAULT_ENCODING = getdefaultencoding()
#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/py3compat.py#L17-L19
def decode(s, encoding=None):
    encoding = encoding or DEFAULT_ENCODING
    return s.decode(encoding, "replace")

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/py3compat.py#L26-L29
def cast_unicode(s, encoding=None):
    if isinstance(s, bytes):
        return decode(s, encoding)
    return s

#Below check completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/timing.py#L23-L67
try:
    import resource
    def clocku():
        """clocku() -> floating point number
        Return the *USER* CPU time in seconds since the start of the process.
        This is done via a call to resource.getrusage, so it avoids the
        wraparound problems in time.clock()."""
        
        return resource.getrusage(resource.RUSAGE_SELF)[0]
    
    def clocks():
        """clocks() -> floating point number
        Return the *SYSTEM* CPU time in seconds since the start of the process.
        This is done via a call to resource.getrusage, so it avoids the
        wraparound problems in time.clock()."""
        
        return resource.getrusage(resource.RUSAGE_SELF)[1]
    
    def clock():
        """clock() -> floating point number
        Return the *TOTAL USER+SYSTEM* CPU time in seconds since the start of
        the process.  This is done via a call to resource.getrusage, so it
        avoids the wraparound problems in time.clock()."""
        
        u,s = resource.getrusage(resource.RUSAGE_SELF)[:2]
        return u+s
    
    def clock2():
        """clock2() -> (t_user,t_system)
        Similar to clock(), but return a tuple of user/system times."""
        return resource.getrusage(resource.RUSAGE_SELF)[:2]
except ImportError:
    # There is no distinction of user/system time under windows, so we just use
    # time.perff_counter() for everything...
    clocku = clocks = clock = time.perf_counter
    def clock2():
        """Under windows, system CPU time can't be measured.
        This just returns perf_counter() and zero."""
        return time.perf_counter(),0.0

#Below check completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/_process_win32.py#L171-L200
try:
    CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
    CommandLineToArgvW.arg_types = [LPCWSTR, POINTER(c_int)]
    CommandLineToArgvW.restype = POINTER(LPCWSTR)
    LocalFree = ctypes.windll.kernel32.LocalFree
    LocalFree.res_type = HLOCAL
    LocalFree.arg_types = [HLOCAL]
    
    def arg_split(commandline, posix=False, strict=True):
        """Split a command line's arguments in a shell-like manner.
        This is a special version for windows that use a ctypes call to CommandLineToArgvW
        to do the argv splitting. The posix parameter is ignored.
        If strict=False, process_common.arg_split(...strict=False) is used instead.
        """
        #CommandLineToArgvW returns path to executable if called with empty string.
        if commandline.strip() == "":
            return []
        if not strict:
            # not really a cl-arg, fallback on _process_common
            return py_arg_split(commandline, posix=posix, strict=strict)
        argvn = c_int()
        result_pointer = CommandLineToArgvW(py3compat.cast_unicode(commandline.lstrip()), ctypes.byref(argvn))
        result_array_type = LPCWSTR * argvn.value
        result = [arg for arg in result_array_type.from_address(ctypes.addressof(result_pointer.contents))]
        retval = LocalFree(result_pointer)
        return result
except AttributeError:
    arg_split = py_arg_split

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/page.py#L81-L124
def _detect_screen_size(screen_lines_def):
    """Attempt to work out the number of lines on the screen.
    This is called by page(). It can raise an error (e.g. when run in the
    test suite), so it's separated out so it can easily be called in a try block.
    """
    TERM = os.environ.get('TERM',None)
    if not((TERM=='xterm' or TERM=='xterm-color') and sys.platform != 'sunos5'):
        # curses causes problems on many terminals other than xterm, and
        # some termios calls lock up on Sun OS5.
        return screen_lines_def
    
    try:
        import termios
        import curses
    except ImportError:
        return screen_lines_def
    
    # There is a bug in curses, where *sometimes* it fails to properly
    # initialize, and then after the endwin() call is made, the
    # terminal is left in an unusable state.  Rather than trying to
    # check every time for this (by requesting and comparing termios
    # flags each time), we just save the initial terminal state and
    # unconditionally reset it every time.  It's cheaper than making
    # the checks.
    try:
        term_flags = termios.tcgetattr(sys.stdout)
    except termios.error as err:
        # can fail on Linux 2.6, pager_page will catch the TypeError
        raise TypeError('termios error: {0}'.format(err)) from err
    
    try:
        scr = curses.initscr()
    except AttributeError:
        # Curses on Solaris may not be complete, so we can't use it there
        return screen_lines_def
    
    screen_lines_real,screen_cols = scr.getmaxyx()
    curses.endwin()
    
    # Restore terminal state in case endwin() didn't.
    termios.tcsetattr(sys.stdout,termios.TCSANOW,term_flags)
    # Now we have what we needed: the screen size in rows/columns
    return screen_lines_real

#Below function copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L1466-L1503
def _format_time(timespan, precision=3):
    if timespan >= 60.0:
        parts = [("d", 60*60*24),("h", 60*60),("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)
    
    units = [u"s", u"ms",u'us',"ns"] # the save value   
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms",u'\xb5s',"ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]
    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return u"%.*g %s" % (precision, timespan * scaling[order], units[order])

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/data.py#L26-L28
def chop(seq, size):
    """Chop a sequence into chunks of the given size."""
    return [seq[i:i+size] for i in range(0,len(seq),size)]

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/page.py#L285-L304
def get_pager_cmd(pager_cmd=None):
    """Return a pager command.
    Makes some attempts at finding an OS-correct one.
    """
    if os.name == 'posix':
        default_pager_cmd = 'less -R'  # -R for color control sequences
    elif os.name in ['nt','dos']:
        default_pager_cmd = 'type'
    
    if pager_cmd is None:
        try:
            pager_cmd = os.environ['PAGER']
        except:
            pager_cmd = default_pager_cmd
    
    if pager_cmd == 'less' and '-r' not in os.environ.get('LESS', '').lower():
        pager_cmd += ' -R'
    
    return pager_cmd

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/page.py#L307-L320
def get_pager_start(pager, start):
    """Return the string for paging files with an offset.
    This is the '+N' argument which less and more (under Unix) accept.
    """
    
    if pager in ['less','more']:
        if start:
            start_string = '+' + str(start)
        else:
            start_string = ''
    else:
        start_string = ''
    return start_string

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/utils/terminal.py#L128-L129
def get_terminal_size(defaultx=80, defaulty=25):
    return _get_terminal_size((defaultx, defaulty))

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/page.py#L57-L79
def page_dumb(strng, start=0, screen_lines=25):
    """Very dumb 'pager' in Python, for when nothing else works.
    Only moves forward, same interface as page(), except for pager_cmd and
    mode.
    """
    if isinstance(strng, dict):
        strng = strng.get('text/plain', '')
    out_ln  = strng.splitlines()[start:]
    screens = chop(out_ln,screen_lines-1)
    if len(screens) == 1:
        print(os.linesep.join(screens[0]))
    else:
        last_escape = ""
        for scr in screens[0:-1]:
            hunk = os.linesep.join(scr)
            print(last_escape + hunk)
            if not page_more():
                return
            esc_list = esc_re.findall(hunk)
            if len(esc_list) > 0:
                last_escape = esc_list[-1]
        print(last_escape + os.linesep.join(screens[-1]))

#Below function completely copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/page.py#L128-L236
def page(strng, start=0, screen_lines=0, pager_cmd=None):
    """Display a string, piping through a pager after a certain length.
    
    strng can be a mime-bundle dict, supplying multiple representations,
    keyed by mime-type.
    The screen_lines parameter specifies the number of *usable* lines of your
    terminal screen (total lines minus lines you need to reserve to show other
    information).
    If you set screen_lines to a number <=0, page() will try to auto-determine
    your screen size and will only use up to (screen_size+screen_lines) for
    printing, paging after that. That is, if you want auto-detection but need
    to reserve the bottom 3 lines of the screen, use screen_lines = -3, and for
    auto-detection without any lines reserved simply use screen_lines = 0.
    If a string won't fit in the allowed lines, it is sent through the
    specified pager command. If none given, look for PAGER in the environment,
    and ultimately default to less.
    If no system pager works, the string is sent through a 'dumb pager'
    written in python, very simplistic.
    """
    
    # for compatibility with mime-bundle form:
    if isinstance(strng, dict):
        strng = strng['text/plain']
    
    # Ugly kludge, but calling curses.initscr() flat out crashes in emacs
    TERM = os.environ.get('TERM','dumb')
    if TERM in ['dumb','emacs'] and os.name != 'nt':
        print(strng)
        return
    # chop off the topmost part of the string we don't want to see
    str_lines = strng.splitlines()[start:]
    str_toprint = os.linesep.join(str_lines)
    num_newlines = len(str_lines)
    len_str = len(str_toprint)
    
    # Dumb heuristics to guesstimate number of on-screen lines the string
    # takes.  Very basic, but good enough for docstrings in reasonable
    # terminals. If someone later feels like refining it, it's not hard.
    numlines = max(num_newlines,int(len_str/80)+1)
    
    screen_lines_def = get_terminal_size()[1]
    
    # auto-determine screen size
    if screen_lines <= 0:
        try:
            screen_lines += _detect_screen_size(screen_lines_def)
        except (TypeError, UnsupportedOperation):
            print(str_toprint)
            return
    
    #print 'numlines',numlines,'screenlines',screen_lines  # dbg
    if numlines <= screen_lines :
        #print '*** normal print'  # dbg
        print(str_toprint)
    else:
        # Try to open pager and default to internal one if that fails.
        # All failure modes are tagged as 'retval=1', to match the return
        # value of a failed system command.  If any intermediate attempt
        # sets retval to 1, at the end we resort to our own page_dumb() pager.
        pager_cmd = get_pager_cmd(pager_cmd)
        pager_cmd += ' ' + get_pager_start(pager_cmd,start)
        if os.name == 'nt':
            if pager_cmd.startswith('type'):
                # The default WinXP 'type' command is failing on complex strings.
                retval = 1
            else:
                fd, tmpname = tempfile.mkstemp('.txt')
                tmppath = Path(tmpname)
                try:
                    os.close(fd)
                    with tmppath.open("wt") as tmpfile:
                        tmpfile.write(strng)
                        cmd = "%s < %s" % (pager_cmd, tmppath)
                    # tmpfile needs to be closed for windows
                    if os.system(cmd):
                        retval = 1
                    else:
                        retval = None
                finally:
                    Path.unlink(tmppath)
        else:
            try:
                retval = None
                # Emulate os.popen, but redirect stderr
                proc = subprocess.Popen(pager_cmd,
                                shell=True,
                                stdin=subprocess.PIPE,
                                stderr=subprocess.DEVNULL
                                )
                pager = os._wrap_close(io.TextIOWrapper(proc.stdin), proc)
                try:
                    pager_encoding = pager.encoding or sys.stdout.encoding
                    pager.write(strng)
                finally:
                    retval = pager.close()
            except IOError as msg:  # broken pipe when user quits
                if msg.args == (32, 'Broken pipe'):
                    retval = None
                else:
                    retval = 1
            except OSError:
                # Other strange problems, sometimes seen in Win2k/cygwin
                retval = 1
        if retval is not None:
            page_dumb(strng,screen_lines=screen_lines)

#Below function has content copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L305-L379
def _run_with_profiler(code, opts, namespace):
        prof = profile.Profile()
        try:
            prof = prof.runctx(code, namespace, namespace)
            sys_exit = ''
        except SystemExit:
            sys_exit = """*** SystemExit exception caught in code being profiled."""
        
        stats = pstats.Stats(prof).strip_dirs().sort_stats(*opts.get('-s', ['time']))
        
        lims = opts.get('-l', [])
        if lims:
            lims, lims_ = [], lims  # rebuild lims with ints/floats/strings
            for lim in lims_:
                try:
                    lims.append(int(lim))
                except ValueError:
                    try:
                        lims.append(float(lim))
                    except ValueError:
                        lims.append(lim)
        
        # Trap output.
        stdout_trap = StringIO()
        stats_stream = stats.stream
        try:
            stats.stream = stdout_trap
            stats.print_stats(*lims)
        finally:
            stats.stream = stats_stream
        
        output = stdout_trap.getvalue()
        output = output.rstrip()
        
        if '-q' not in opts:
            page(output)
        print(sys_exit, end=' ')
        
        dump_file = opts.get('-D', [''])[0]
        text_file = opts.get('-T', [''])[0]
        if dump_file:
            prof.dump_stats(dump_file)
            print(
                f"\n*** Profile stats marshalled to file {repr(dump_file)}.{sys_exit}"
            )
        if text_file:
            pfile = Path(text_file)
            pfile.touch(exist_ok=True)
            pfile.write_text(output)
            
            print(
                f"\n*** Profile printout saved to text file {repr(text_file)}.{sys_exit}"
            )
        
        if '-r' in opts:
            return stats
        
        return None

#Below class copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L60-L110
class TimeitResult(object):
    """
    Object returned by the timeit magic with info about the run.
    Contains the following attributes :
    loops: (int) number of loops done per measurement
    repeat: (int) number of times the measurement has been repeated
    best: (float) best execution time / number
    all_runs: (list of float) execution time of each run (in s)
    compile_time: (float) time of statement compilation (s)
    """
    def __init__(self, loops, repeat, best, worst, all_runs, compile_time, precision):
        self.loops = loops
        self.repeat = repeat
        self.best = best
        self.worst = worst
        self.all_runs = all_runs
        self.compile_time = compile_time
        self._precision = precision
        self.timings = [ dt / self.loops for dt in all_runs]
    
    @property
    def average(self):
        return math.fsum(self.timings) / len(self.timings)
    
    @property
    def stdev(self):
        mean = self.average
        return (math.fsum([(x - mean) ** 2 for x in self.timings]) / len(self.timings)) ** 0.5
    
    def __str__(self):
        pm = '+-'
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            try:
                u'\xb1'.encode(sys.stdout.encoding)
                pm = u'\xb1'
            except:
                pass
        return (
            u"{mean} {pm} {std} per loop (mean {pm} std. dev. of {runs} run{run_plural}, {loops} loop{loop_plural} each)"
                .format(
                    pm = pm,
                    runs = self.repeat,
                    loops = self.loops,
                    loop_plural = "" if self.loops == 1 else "s",
                    run_plural = "" if self.repeat == 1 else "s",
                    mean = _format_time(self.average, self._precision),
                    std = _format_time(self.stdev, self._precision))
                )

#Below class copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L117-L139
class TimeitTemplateFiller(ast.NodeTransformer):
    """Fill in the AST template for timing execution.
    This is quite closely tied to the template definition, which is in
    :meth:`ExecutionMagics.timeit`.
    """
    def __init__(self, ast_setup, ast_stmt):
        self.ast_setup = ast_setup
        self.ast_stmt = ast_stmt
    
    def visit_FunctionDef(self, node):
        "Fill in the setup statement"
        self.generic_visit(node)
        if node.name == "inner":
            node.body[:1] = self.ast_setup.body
        
        return node
    
    def visit_For(self, node):
        "Fill in the statement to be timed"
        if getattr(getattr(node.body[0], 'value', None), 'id', None) == 'stmt':
            node.body = self.ast_stmt.body
        return node

#Below class copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L142-L167
class Timer(timeit.Timer):
    """Timer class that explicitly uses self.inner
    
    which is an undocumented implementation detail of CPython,
    not shared by PyPy.
    """
    # Timer.timeit copied from CPython 3.4.2
    def timeit(self, number=timeit.default_number):
        """Time 'number' executions of the main statement.
        To be precise, this executes the setup statement once, and
        then returns the time it takes to execute the main statement
        a number of times, as a float measured in seconds.  The
        argument is the number of times through the loop, defaulting
        to one million.  The main statement, the setup statement and
        the timer function to be used are passed to the constructor.
        """
        it = itertools.repeat(None, number)
        gcold = gc.isenabled()
        gc.disable()
        try:
            timing = self.inner(it, self.timer)
        finally:
            if gcold:
                gc.enable()
        return timing

#Below function copied from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magic.py#L575-L668
def parse_options(self, arg_str, opt_str, *long_opts, **kw):
        """Parse options passed to an argument string.
        The interface is similar to that of getopt.getopt, but it
        returns a dictionary with the options as keys
        and the stripped argument string still as a string.
        arg_str is quoted as a true sys.argv vector by using shlex.split.
        This allows us to easily expand variables, glob files, quote
        arguments, etc.
        Parameters
        ----------
        arg_str : str
          The arguments to parse.
        opt_str : str
          The options specification.
        mode : str, default 'string'
          If given as 'list', the argument string is returned as a list (split
          on whitespace) instead of a string.
        list_all : bool, default False
          Put all option values in lists. Normally only options
          appearing more than once are put in a list.
        posix : bool, default True
          Whether to split the input line in POSIX mode or not, as per the
          conventions outlined in the :mod:`shlex` module from the standard
          library.
        """
        
        mode = kw.get('mode','string')
        if mode not in ['string','list']:
            raise ValueError('incorrect mode given: %s' % mode)
        # Get options
        list_all = kw.get('list_all',0)
        posix = kw.get('posix', os.name == 'posix')
        strict = kw.get('strict', True)
        
        preserve_non_opts = kw.get("preserve_non_opts", False)
        remainder_arg_str = arg_str
        
        # Check if we have more than one argument to warrant extra processing:
        odict = {}  # Dictionary with options
        args = arg_str.split()
        if len(args) >= 1:
            # If the list of inputs only has 0 or 1 thing in it, there's no
            # need to look for options
            argv = arg_split(arg_str, posix, strict)
            # Do regular option processing
            try:
                opts,args = getopt.getopt(argv, opt_str, long_opts)
            except GetoptError as e:
                raise UsageError(
                    '%s ( allowed: "%s" %s)' % (e.msg, opt_str, " ".join(long_opts))
                ) from e
            for o, a in opts:
                if mode == "string" and preserve_non_opts:
                    # remove option-parts from the original args-string and preserve remaining-part.
                    # This relies on the arg_split(...) and getopt(...)'s impl spec, that the parsed options are
                    # returned in the original order.
                    remainder_arg_str = remainder_arg_str.replace(o, "", 1).replace(
                        a, "", 1
                    )
                if o.startswith("--"):
                    o = o[2:]
                else:
                    o = o[1:]
                try:
                    odict[o].append(a)
                except AttributeError:
                    odict[o] = [odict[o],a]
                except KeyError:
                    if list_all:
                        odict[o] = [a]
                    else:
                        odict[o] = a
        
        # Prepare opts,args for return
        if mode == 'string':
            if preserve_non_opts:
                args = remainder_arg_str.lstrip()
            else:
                args = " ".join(args)
        
        return odict,args

#Implementation of %prun from https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L184-L303
def magic_prun(parameter_s=''):
    """
    Usage:
    magiccmds.prun("[option] <stmt-string or expression>")
    """
    opts, arg_str = parse_options(parameter_s, 'D:l:rs:T:q')
    arg_str = '\n'.join(arg_str)
    optdict = {}
    for key, value in opts:
        if key in optdict:
            optdict[key].append(value)
        else:
            optdict[key] = [value]
    
    opts = optdict
    return _run_with_profiler(arg_str, opts, globals())

default_timert = timeit.default_timer
default_repeatt = timeit.default_repeat
timet = time.time
#Implementation of %timeit from https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L1002-L1189
def magic_timeit(line='', local_ns=None):
    """
    Usage:
    magiccmds.timeit("[option] (-s <setup-string or expression>)* <stmt-string or expression>", namespace)
    """
    opts, stmt = parse_options(line, 'n:r:s:tcp:qo')
    if not stmt:
        return
    
    timefunc = default_timert
    setupstmt = '\n'.join([x[1] for x in opts if x[0] == '-s'])
    stmt = '\n'.join(stmt)
    opts = dict(opts)
    number = int(opts.get("-n", 0))
    default_repeat = 7 if default_repeatt < 7 else default_repeatt
    repeat = int(opts.get("-r", default_repeat))
    precision = int(opts.get("-p", 3))
    quiet, return_result = '-q' in opts, '-o' in opts
    if '-c' in opts:
        timefunc = clock
    elif 't' in opts:
        timefunc = timet
    
    timer = Timer(timer=timefunc)
    valid = True
    try:
        setupstmt = compile(setupstmt, "<timeit-magic-setup>", "exec", ast.PyCF_ONLY_AST)
    except SyntaxError:
        valid = False
        traceback.print_exc()
    
    try:
        stmt = compile(stmt, "<timeit-magic-stmt>", "exec", ast.PyCF_ONLY_AST)
    except SyntaxError:
        if valid is not False: valid = False
        traceback.print_exc()
    
    if valid:
        timeit_ast_template = ast.parse('def inner(_it, _timer):\n'
                                        '    setupstmt\n'
                                        '    _t0 = _timer()\n'
                                        '    for _i in _it:\n'
                                        '        stmt\n'
                                        '    _t1 = _timer()\n'
                                        '    return _t1 - _t0\n')
        timeit_ast = TimeitTemplateFiller(setupstmt, stmt).visit(timeit_ast_template)
        timeit_ast = ast.fix_missing_locations(timeit_ast)
        tc_min = 0.1
        t0 = clock()
        code = compile(timeit_ast, "<timeit-magic>", "exec")
        tc = clock()-t0
        ns = {}
        glob = (thestuff:=stack()[1][0]).f_globals|thestuff.f_locals
        conflict_globs = {}
        if local_ns:
            for var_name, var_val in glob.items():
                if var_name in local_ns:
                    conflict_globs[var_name] = var_val
            glob.update(local_ns)
        
        exec(code, glob, ns)
        timer.inner = ns["inner"]
        if number == 0:
            for index in range(10):
                number = 10**index
                time_number = timer.timeit(number)
                if time_number >= .2:
                    break
        
        all_runs = timer.repeat(repeat, number)
        best = min(all_runs)/number
        worst = max(all_runs)/number
        timeit_result = TimeitResult(number, repeat, best, worst, all_runs, tc, precision)
        if conflict_globs:
            glob.update(conflict_globs)
        
        if not quiet:
            if worst > 4 * best and best > 0 and worst > 1e-6:
                print("The slowest run took {:.2f} times longer than the "
                      "fastest. This could mean that an intermediate result "
                      "is being cached.".format(worst/best))
            
            print(timeit_result)
            if tc > tc_min:
                print("Compiler time: {:.2f} s".format(tc))
        if return_result:
            return timeit_result

#Below check copied completely from IPython source code, https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L46-L52
if sys.version_info > (3,8):
    from ast import Module
else :
    # mock the new API, ignore second argument
    # see https://github.com/ipython/ipython/issues/11590
    from ast import Module as OriginalModule
    Module = lambda nodelist, type_ignores: OriginalModule(nodelist)

#Implementation of %time from https://github.com/ipython/ipython/blob/master/IPython/core/magics/execution.py#L1195-L1338
def magic_time(line='', local_ns=None):
    """
    Usage:
    magiccmds.timeit("<stmt-string or expression>", namespace)
    """
    tp_min = 0.1
    t0 = clock()
    expr_ast = ast.parse(line)
    tp = clock()-t0
    expr_ast = ast.fix_missing_locations(expr_ast)
    tc_min = 0.1
    expr_val = None
    if len(expr_ast.body)==1 and isinstance(expr_ast.body[0], ast.Expr):
        mode = 'eval'
        source = '<timed eval>'
        expr_ast = ast.Expression(expr_ast.body[0].value)
    else:
        mode = 'exec'
        source = '<timed exec>'
        if len(expr_ast.body) > 1 and isinstance(expr_ast.body[-1], ast.Expr):
            expr_val = expr_ast.body[-1]
            expr_ast = expr_ast.body[:-1]
            expr_ast = Module(expr_ast, [])
    
    t0 = clock()
    code = compile(expr_ast, source, mode)
    tc = clock()-t0
    glob = (thestuff:=stack()[1][0]).f_globals|thestuff.f_locals
    wtime = timet
    wall_st = wtime()
    if mode == 'eval':
        st = clock2()
        try:
            exec(code, glob, local_ns)
            out = None
            if expr_val is not None:
                code_2 = compile(expr_val, source, 'eval')
                out = eval(code_2, glob, local_ns)
        except:
            traceback.print_exc()
            return
        
        end = clock2()
    
    wall_end = wtime()
    wall_time = wall_end-wall_st
    cpu_user = end[0]-st[0]
    cpu_sys = end[1]-st[1]
    cpu_tot = cpu_user+cpu_sys
    if sys.platform != 'win32':
        print("CPU times: user {}, sys {}, total {}" \
                  .format(_format_time(cpu_user), _format_time(cpu_sys), _format_time(cpu_tot)))
    
    print("Wall time: {}".format(_format_time(wall_time)))
    if tc > tc_min:
        print("Compiler : {}".format(_format_time(tc)))
    
    if tp > tp_min:
        print("Parser   : {}".format(_format_time(tp)))
    
    return out

