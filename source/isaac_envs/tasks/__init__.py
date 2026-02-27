# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

import os

# When running unit tests that don't need full Isaac Lab / USD, we allow
# callers to skip importing tasks (and thus avoid importing heavy deps).
if os.environ.get("ISAAC_ENVS_SKIP_TASK_IMPORTS") != "1":
    from isaaclab_tasks.utils import import_packages

    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)
