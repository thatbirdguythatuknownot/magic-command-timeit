import timeit, time, gc, itertools, shlex, getopt, ast, traceback, sys, math
try:
    import resource
    def clock():
        u,s = resource.getrusage(resource.RUSAGE_SELF)[:2]
        return u+s
except ImportError:
    clock = time.perf_counter

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

def magic_timeit(totime, local_ns=None):
    opts, stmt = getopt.getopt(shlex.split(totime), 'n:r:s:tcp:qo')
    timefunc = timeit.default_timer
    setupstmt = '\n'.join([x[1] for x in opts if x[0] == '-s'])
    stmt = '\n'.join(stmt)
    opts = dict(opts)
    number = int(opts.get("n", 0))
    default_repeat = 7 if timeit.default_repeat < 7 else timeit.default_repeat
    repeat = int(opts.get("r", default_repeat))
    precision = int(opts.get("p", 3))
    quiet, return_result = 'q' in opts, 'o' in opts
    if 'c' in opts:
        timefunc = clock
    elif 't' in opts:
        timefunc = time.time
    
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
        glob = globals()
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

