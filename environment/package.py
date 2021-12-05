#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"defining a class for package"
class Package(object):
    def __init__(self, source: int, destination: int, size: int, path_obj):
        """
        Creates a Package Object
        :param source: Source Node
        :param destination: Destination Node
        :param size: Size of the package
        """
        if source < 0:
            raise ValueError("Error in Package Init! Source must be non-negative")
        if destination < 0:
            raise ValueError("Error in Package Init! Destination must be non-negative")
        if size <= 0:
            raise ValueError("Error in Package Init! Size must be positive")
        
        # package info
        self.source = source              # source of the package
        self.destination = destination    # destination of the package
        self.size = size                  # size of the request
        self.path_obj = path_obj
        # package state
        self.location = source            # current location

