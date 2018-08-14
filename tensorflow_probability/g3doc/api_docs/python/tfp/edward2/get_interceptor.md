<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.get_interceptor" />
</div>

# tfp.edward2.get_interceptor

``` python
tfp.edward2.get_interceptor()
```

Returns the top-most (last) interceptor on the thread's stack.

The bottom-most (first) interceptor in the stack is a function which takes
`f, *args, **kwargs` as input and returns `f(*args, **kwargs)`. It is the
default if no `interception` contexts have been entered.