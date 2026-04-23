# DEV Changes-
```bash
diff --git a/franka_hw/include/franka_hw/resource_helpers.h b/franka_hw/include/franka_hw/resource_helpers.h
index f2231ba..fec7451 100644
--- a/franka_hw/include/franka_hw/resource_helpers.h
+++ b/franka_hw/include/franka_hw/resource_helpers.h
@@ -10,6 +10,7 @@
 #include <hardware_interface/controller_info.h>
 
 #include <franka_hw/control_mode.h>
+#include <cstdint>
 
 namespace franka_hw {
```

# ROS integration for Franka Emika research robots

[![Build Status][travis-status]][travis]

See the [Franka Control Interface (FCI) documentation][fci-docs] for more information.

## License

All packages of `franka_ros` are licensed under the [Apache 2.0 license][apache-2.0].

[apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0.html
[fci-docs]: https://frankaemika.github.io/docs
[travis-status]: https://travis-ci.org/frankaemika/franka_ros.svg?branch=kinetic-devel
[travis]: https://travis-ci.org/frankaemika/franka_ros
