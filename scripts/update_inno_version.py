#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新 Inno Setup 文件中的 AppVersion
从 constants.py 读取 APP_VERSION，获取当前 Git 提交哈希，
组合成类似 4.1.0-hash 的格式并更新到 inno/SuperPicky.iss
"""

import os
import sys
import subprocess


def get_git_commit_hash():
    """
    获取当前 Git 提交哈希
    
    Returns:
        str: 7位 Git 提交哈希
    """
    try:
        # 执行 git 命令获取当前提交哈希
        result = subprocess.run(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"获取 Git 提交哈希失败: {result.stderr}")
            return "unknown"
    except Exception as e:
        print(f"获取 Git 提交哈希出错: {e}")
        return "unknown"


def read_app_version():
    """
    从 constants.py 读取 APP_VERSION
    
    Returns:
        str: 应用版本号
    """
    constants_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'constants.py'
    )
    
    try:
        with open(constants_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('APP_VERSION ='):
                    # 提取版本号，处理引号
                    version = line.split('=', 1)[1].strip()
                    if version.startswith('"') and version.endswith('"'):
                        return version[1:-1]
                    elif version.startswith("'") and version.endswith("'"):
                        return version[1:-1]
                    return version
        print("未找到 APP_VERSION")
        return "0.0.0"
    except Exception as e:
        print(f"读取 constants.py 出错: {e}")
        return "0.0.0"


def update_inno_version():
    """
    更新 inno/SuperPicky.iss 中的 AppVersion
    """
    # 获取版本号和哈希
    app_version = read_app_version()
    commit_hash = get_git_commit_hash()
    
    # 组合版本字符串
    new_version = f"{app_version} {commit_hash}"
    print(f"更新版本为: {new_version}")
    
    # 定位 inno 文件
    inno_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'inno', 'SuperPicky.iss'
    )
    
    try:
        # 读取文件内容
        with open(inno_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换 AppVersion
        import re
        updated_content = re.sub(
            r'AppVersion=.+',
            f'AppVersion={new_version}',
            content
        )
        
        # 写回文件
        with open(inno_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"成功更新 {inno_path}")
        return True
    except Exception as e:
        print(f"更新 inno 文件出错: {e}")
        return False


if __name__ == "__main__":
    success = update_inno_version()
    sys.exit(0 if success else 1)
