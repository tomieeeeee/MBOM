from threading import Thread
import threading
import multiprocessing

class MyThread(threading.Thread):
    def __init__(self, func, args, kwargs=None, name=''):
        threading.Thread.__init__(self)
        self.func = func
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self.kwargs == None:
            self.res = self.func(*self.args)
        else:
            print(self.args)
            print(self.kwargs)

            self.res = self.func(*self.args, self.kwargs)

    def getResult(self):
        return self.res

def mt_tool(objs, fn_name, args_group, kwargs_group=None):
    '''
    multi-threading for object.function
    class of obj must have eval function, such as
        def eval(self, fn_name):
            return eval("self." + fn_name)
    :param objs:  [num_trj]
    :param fn_name: function name of obj
    :param args_group:  list of function param  [(p1,p2...), (p1,p2...) ...]
    :return: list of function's return
    '''
    assert type(args_group) is list and type(args_group[0]) is tuple, "args_group type error, should be group of list and tuple, such as [(p1,p2...), (p1,p2...) ...]"
    assert len(objs) == len(args_group), "length must be same for objs and args_group"
    t_l = []
    if kwargs_group == None:
        for obj, args in zip(objs, args_group):
            if isinstance(obj, object):
                fn = getattr(obj, fn_name)
            else:
                fn = fn_name
            #t = Thread(target=fn, args=(args,))
            t = MyThread(func=fn, args=args)
            t.start()
            t_l.append(t)
    else:
        for obj, args, kwargs in zip(objs, args_group, kwargs_group):
            if isinstance(obj, object):
                fn = getattr(obj, fn_name)
            else:
                fn = fn_name
            #t = Thread(target=fn, args=(args,))
            t = MyThread(func=fn, args=args, kwargs=kwargs)
            t.start()
            t_l.append(t)
    for t in t_l:
        t.join()
    res = [t.getResult() for t in t_l]
    return res

if __name__ == "__main__":
    def fn(name):
        for _ in range(100000):
            print("{}".format(name))

    class A():
        def __init__(self):
            pass

        def action(self, name):
            for _ in range(10000):
                print("{}".format(name))
            return "123"

        def eval(self, fn_name):
            return eval("self." + fn_name)

    objs = [A() for i in range(100)]
    args_group = [str(i) for i in range(100)]
    print(mt_tool(objs, "action", args_group))

    # t_l = []
    # for i in range(10):
    #     t = Thread(target=fn, args=(i,))
    #     t.start()
    #     t_l.append(t)
    # for t in t_l:
    #     t.join()
    # while True:
    #     print("end")