# Test to see if kernels compilation work fine in the tested device, and if binary caching works

@GPU
Feature: check GPU compatibility

    Scenario: check GPU compatibility
        When I launch videostitch-cmd with "--check_gpu_compatibility"
        Then I expect the command to succeed
	
        When I launch videostitch-cmd with "-d 0 --check_gpu_compatibility "
        Then I expect the command to succeed

        When I launch videostitch-cmd with "-d 1 --check_gpu_compatibility "
        Then I expect the command to succeed

#Test the cache
        When I launch videostitch-cmd with "-d 0 --check_gpu_compatibility "
        Then I expect program compilation to take less than 10 seconds

        When I launch videostitch-cmd with "-d 1 --check_gpu_compatibility "
        Then I expect program compilation to take less than 10 seconds
