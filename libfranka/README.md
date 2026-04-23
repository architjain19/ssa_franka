# DEV Changes-
```bash
diff --git a/include/franka/control_tools.h b/include/franka/control_tools.h
index dc5017c3..0bd2c30b 100644
--- a/include/franka/control_tools.h
+++ b/include/franka/control_tools.h
@@ -4,6 +4,7 @@
 
 #include <array>
 #include <cmath>
+#include <string>
 
 /**
  * @file control_tools.h
diff --git a/src/control_types.cpp b/src/control_types.cpp
index 046062e5..0c6bd58a 100644
--- a/src/control_types.cpp
+++ b/src/control_types.cpp
@@ -3,6 +3,7 @@
 #include <type_traits>
 
 #include <franka/control_types.h>
+#include <stdexcept>
 
 namespace franka {
```

# libfranka: C++ library for Franka Emika research robots

[![Build Status][travis-status]][travis]
[![codecov][codecov-status]][codecov]

With this library, you can control research versions of Franka Emika robots. See the [Franka Control Interface (FCI) documentation][fci-docs] for more information about what `libfranka` can do and how to set it up. The [generated API documentation][api-docs] also gives an overview of its capabilities.

## License

`libfranka` is licensed under the [Apache 2.0 license][apache-2.0].

[apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0.html
[api-docs]: https://frankaemika.github.io/libfranka
[fci-docs]: https://frankaemika.github.io/docs
[travis-status]: https://travis-ci.org/frankaemika/libfranka.svg?branch=master
[travis]: https://travis-ci.org/frankaemika/libfranka
[codecov-status]: https://codecov.io/gh/frankaemika/libfranka/branch/master/graph/badge.svg
[codecov]: https://codecov.io/gh/frankaemika/libfranka
