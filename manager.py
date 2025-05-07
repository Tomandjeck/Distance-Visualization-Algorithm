import subprocess
import time

count=0



# 定义最大运行次数
max_runs = 100
run_count = 0

# 无限循环，监控程序
while True:
    try:
        # 要运行的目标程序及其参数
        program_command = ["python", "./model.py", "--arg1", str(count)]

        # 启动程序并等待完成
        print(f"Running the program ({run_count + 1}/{max_runs})...")
        process = subprocess.run(program_command, check=True)

        # 增加运行次数
        run_count += 1
        count+=1
        # 如果达到运行次数限制，重新初始化运行次数
        if run_count >= max_runs:
            print("Program reached 10 runs, restarting...")
            run_count = 0  # 重置计数器
            time.sleep(10)  # 可选：暂停几秒钟，避免频繁重启

    except subprocess.CalledProcessError as e:
        # 如果程序运行出现错误，打印错误信息
        print(f"Program terminated with an error: {e}")
        break
    except KeyboardInterrupt:
        # 如果用户按下 Ctrl+C，安全退出
        print("Terminated by user.")
        break
