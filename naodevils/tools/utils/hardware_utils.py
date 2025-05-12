from utils import *


def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        import re
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        import subprocess
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        import re
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            import subprocess
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')


def disable_gpu():
    from utils.setup_tensorflow_utils import tf
    try:
        # Disable all GPUS
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except Exception as e:
        eprint(str(e))


def set_gpu_and_cpu_usage():
    """
    sets the gpu usage memory fraction and the number of parallel cpu threads
    """
    from utils.setup_tensorflow_utils import tf
    if DEBUGGING:
        print_seperator()
        print("[set_GPU_and_CPU_usage] Before:")
        print("cpu_intra_op_parallelism_threads: " + str(tf.config.threading.get_intra_op_parallelism_threads()))
        print("cpu_inter_op_parallelism_threads: " + str(tf.config.threading.get_inter_op_parallelism_threads()))
        # print("gpu_per_process_memory_fraction: " + str(tf.config.gpu.get_per_process_memory_fraction()))
        # print("gpu_per_process_memory_growth: " + str(tf.config.gpu.get_per_process_memory_growth()))

    try:
        tf.config.threading.set_intra_op_parallelism_threads(available_cpu_count())
        tf.config.threading.set_inter_op_parallelism_threads(available_cpu_count())
        tf.config.set_soft_device_placement(True)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                eprint(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                eprint(e)
        tf.debugging.set_log_device_placement(True)
        # if DEBUGGING:
        #     tf.debugging.set_log_device_placement(True)
        # else:
        #     tf.debugging.set_log_device_placement(False)
        # see https://www.tensorflow.org/beta/guide/using_gpu?hl=en
        # tf.config.gpu.set_per_process_memory_fraction(FLAGS.gpu_usage)
        # if FLAGS.gpu_usage >= 1.0:
        #     tf.config.gpu.set_per_process_memory_growth(True)

    except RuntimeError as re:
        print("Not the first launch of the programm so skipping!")
        print(str(re))

    if DEBUGGING:
        print_seperator()
        print("[set_GPU_and_CPU_usage] After:")
        print("cpu_intra_op_parallelism_threads: " + str(tf.config.threading.get_intra_op_parallelism_threads()))
        print("cpu_inter_op_parallelism_threads: " + str(tf.config.threading.get_inter_op_parallelism_threads()))
        # print("gpu_per_process_memory_fraction: " + str(tf.config.gpu.get_per_process_memory_fraction()))
        # print("gpu_per_process_memory_growth: " + str(tf.config.gpu.get_per_process_memory_growth()))
        print_seperator()
