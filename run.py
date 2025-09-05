from pathlib import Path
from src.edot_ET_new import main  # 绝对导入主函数

ROOT = Path(__file__).parent

if __name__ == '__main__':
    main(ROOT, zeta=0.002)  # 调用原edot.py中的main()
