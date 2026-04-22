"""Web服务器启动脚本"""

import sys
import os
from src.web_server import run_web_server


def main():
    """主函数"""
    # 确保工作目录正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
    
    # 运行Web服务器
    print("启动带钢浪形检测Web服务器...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    
    try:
        run_web_server()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
