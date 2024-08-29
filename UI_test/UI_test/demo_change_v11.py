import sys
sys.path.append("D:/document/3v3")
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QLabel, QPushButton, QSpinBox, QComboBox, QApplication, QMainWindow, \
    QPlainTextEdit
from PyQt5.QtCore import QTimer, QRect, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from env_wapper.mpe.make_env import make_env
from env_wapper.simple_tag.simple_tag import Simple_Tag
from policy.MBAM import MBAM
from policy.MBAM_OM_MH import MBAM_OM_MH
from policy.MBAM_MH import MBAM_MH
from baselines.PPO import PPO, PPO_Buffer
from baselines.PPO_MH import PPO_MH, PPO_MH_Buffer
from baselines.PPO_OM_MH import PPO_OM_MH, PPO_OM_MH_Buffer
from env_wapper.simple_tag.simple_tag import Simple_Tag
import argparse
import time
from matplotlib import image as mpimg

class all_data():
    def __init__(self):
        # 数量
        self.oppo_num = 6
        self.agent_num = 6
        # 对手类型
        self.oppo_policy = None
        # 对手编号
        self.oppo_idx = "3"
        # 对手动作
        self.oppo_action = ["-","↑","↓","←","→","-"]

        #对战模式
        self.battle_mode = None

        # 航程
        self.hangcheng = "0"

        # 燃油
        self.oil_cost = 0

        # 战损比
        self.zhansunbi = 0.1

        # 解算时间
        self.cac_time = 0.01
       
        # 我方击落对手数量
        self.our_shots = 0

        # 对手击落我方数量
        self.their_shots = 0
    
    def cac_hangcheng(self, action):
        hangcheng_sum = 0.1
        for i in action:
            hangcheng_sum += float(i)  # 确保action中的值被转换为float
        return hangcheng_sum

    def cac_oil(self, agent_num):
        if agent_num == 0:
            return 0
        oil = 1 / agent_num
        return oil

all_data1=all_data()




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 设置图形显示窗口
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(250, 20, 555, 555))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # 添加 matplotlib FigureCanvas 用于绘图
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(QtCore.QRect(250, 20, 555, 555))
        self.canvas.setParent(self.centralwidget)

        # 标签和输入控件设置
        self.setup_labels_and_inputs()

        self.range_value = QLabel(self.centralwidget)
        self.range_value.setGeometry(QtCore.QRect(122, 230, 60, 20))
        self.range_value.setObjectName("range_value")
        self.range_value.setStyleSheet("border: 1px solid black;")

        self.fuel_value = QLabel(self.centralwidget)
        self.fuel_value.setGeometry(QtCore.QRect(122, 250, 60, 20))
        self.fuel_value.setObjectName("fuel_value")
        self.fuel_value.setStyleSheet("border: 1px solid black;")

        # 设置对手行为预测标签
        self.setup_adversary_action_labels()
        self.update_display_from_data()  # 更新显示信息

        # 添加终端输出窗口，移到右侧
        self.terminal_output = QPlainTextEdit(self.centralwidget)
        self.terminal_output.setGeometry(QtCore.QRect(820, 200, 255, 300))
        self.terminal_output.setObjectName("terminal_output")
        self.terminal_output.setReadOnly(True)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def update_display_from_data(self):
        self.range_value.setText(str(all_data1.hangcheng))
        self.fuel_value.setText(str(all_data1.oil_cost))
        self.calc_time_value.setText(str(all_data1.cac_time))
        self.our_shots_value.setText(str(all_data1.our_shots))
        self.their_shots_value.setText(str(all_data1.their_shots))
        for i, action_label in enumerate(self.adversary_action_outputs):
            action_label.setText(all_data1.oppo_action[i])

    def setup_labels_and_inputs(self):
        # 标签：对手类型
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(810, 122, 80, 20))
        self.label.setObjectName("label")

        # 标签：我方数量
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(810, 60, 80, 20))
        self.label_2.setObjectName("label_2")

        # 标签：对手数量
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(810, 90, 80, 20))
        self.label_3.setObjectName("label_3")

        # 文本：预设
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(810, 30, 60, 20))
        self.label_4.setObjectName("label_4")

        # 按钮：开始
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(900, 500, 75, 30))
        self.pushButton.setObjectName("pushButton")

        # 按钮：结束
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(900, 530, 75, 30))
        self.pushButton_2.setObjectName("pushButton_2")

        # 按钮：载入预设
        self.pushButton_3 = QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(900, 560, 80, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.load_preset)  # 连接点击事件

        # 按钮：推演
        self.pushButton_4 = QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(90, 270, 80, 30))
        self.pushButton_4.setObjectName("pushButton_4")

        # 策略类型选择
        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(900, 120, 120, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("PPO")
        self.comboBox.addItem("Meta-PG")

        # 标签：对战模式
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(810, 154, 80, 20))
        self.label_5.setObjectName("label_5")

        # 模式选择（空对空，空对地）
        self.comboBox_2 = QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(900, 154, 120, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("空对空")
        self.comboBox_2.addItem("空对地")

        # 数值输入控件
        self.agent_num = QSpinBox(self.centralwidget)
        self.agent_num.setGeometry(QtCore.QRect(900, 60, 42, 22))
        self.agent_num.setObjectName("agent_num")

        self.adversary_num = QSpinBox(self.centralwidget)
        self.adversary_num.setGeometry(QtCore.QRect(900, 90, 42, 22))
        self.adversary_num.setObjectName("adversary_num")

        # 航程和燃油
        '''
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(60, 230, 60, 20))
        self.label_6.setObjectName("label_6")

        self.range_value = QLabel(self.centralwidget)
        self.range_value.setGeometry(QtCore.QRect(122, 230, 60, 20))
        self.range_value.setObjectName("range_value")
        self.range_value.setStyleSheet("border: 1px solid black;")

        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(60, 250, 60, 20))
        self.label_7.setObjectName("label_7")

        self.fuel_value = QLabel(self.centralwidget)
        self.fuel_value.setGeometry(QtCore.QRect(122, 250, 60, 20))
        self.fuel_value.setObjectName("fuel_value")
        self.fuel_value.setStyleSheet("border: 1px solid black;")
        '''
        # 航程和燃油
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(60, 230, 60, 20))
        self.label_6.setObjectName("label_6")

        self.range_value = QLabel(self.centralwidget)
        self.range_value.setGeometry(QtCore.QRect(122, 230, 60, 20))
        self.range_value.setObjectName("range_value")
        self.range_value.setStyleSheet("border: 1px solid black;")

        self.range_unit = QLabel(self.centralwidget)
        self.range_unit.setGeometry(QtCore.QRect(182, 230, 40, 20))
        self.range_unit.setObjectName("range_unit")
        self.range_unit.setText("千米")

        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(60, 250, 60, 20))
        self.label_7.setObjectName("label_7")

        self.fuel_unit = QLabel(self.centralwidget)
        self.fuel_unit.setGeometry(QtCore.QRect(182, 250, 40, 20))
        self.fuel_unit.setObjectName("fuel_unit")
        self.fuel_unit.setText("千克")

        self.fuel_value = QLabel(self.centralwidget)
        self.fuel_value.setGeometry(QtCore.QRect(122, 250, 60, 20))
        self.fuel_value.setObjectName("fuel_value")
        self.fuel_value.setStyleSheet("border: 1px solid black;")

        # 对手编号
        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(50, 85, 80, 30))
        self.label_8.setObjectName("label_8")

        self.adversary_idx = QSpinBox(self.centralwidget)
        self.adversary_idx.setGeometry(QtCore.QRect(130, 90, 42, 22))
        self.adversary_idx.setObjectName("adversary_idx")

        # 设置对手行为预测标签
        self.setup_adversary_action_labels()

        # 添加解算时间标签和输出框
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(30, 210, 100, 20))
        self.label_9.setObjectName("label_9")
        self.label_9.setText("解算时间")

        self.calc_time_value = QLabel(self.centralwidget)
        self.calc_time_value.setGeometry(QtCore.QRect(122, 210, 60, 20))
        self.calc_time_value.setObjectName("calc_time_value")
        self.calc_time_value.setStyleSheet("border: 1px solid black;")

        self.calc_time_unit = QLabel(self.centralwidget)
        self.calc_time_unit.setGeometry(QtCore.QRect(182, 210, 30, 20))
        self.calc_time_unit.setObjectName("calc_time_unit")
        self.calc_time_unit.setText("秒")

        # 添加我方击落对手数量和输出框
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(60, 500, 120, 20))
        self.label_10.setObjectName("label_10")
        self.label_10.setText("我方击落对手")

        self.our_shots_value = QLabel(self.centralwidget)
        self.our_shots_value.setGeometry(QtCore.QRect(180, 500, 60, 20))
        self.our_shots_value.setObjectName("our_shots_value")
        self.our_shots_value.setStyleSheet("border: 1px solid black;")

        # 添加对手击落我方数量和输出框
        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(60, 480, 120, 20))
        self.label_11.setObjectName("label_11")
        self.label_11.setText("对手击落我方")

        self.their_shots_value = QLabel(self.centralwidget)
        self.their_shots_value.setGeometry(QtCore.QRect(180, 480, 60, 20))
        self.their_shots_value.setObjectName("their_shots_value")
        self.their_shots_value.setStyleSheet("border: 1px solid black;")

    def setup_adversary_action_labels(self):
        self.adversary_action_labels = []
        self.adversary_action_outputs = []
        for i in range(6):
            label = QLabel(self.centralwidget)
            label.setGeometry(QtCore.QRect(50, 300 + i * 30, 120, 20))
            label.setObjectName(f"adversary_action_label_{i + 1}")
            label.setText(f"对手{i + 1}行为预测")
            self.adversary_action_labels.append(label)

            output = QLabel(self.centralwidget)
            output.setGeometry(QtCore.QRect(180, 300 + i * 30, 60, 20))
            output.setObjectName(f"adversary_action_output_{i + 1}")
            output.setStyleSheet("border: 1px solid black;")
            self.adversary_action_outputs.append(output)

    def load_preset(self):
        # 获取当前设置
        adversary_num = self.adversary_num.value()
        all_data1.oppo_num = adversary_num
        agent_num = self.agent_num.value()
        all_data1.agent_num = agent_num
        # 对手策略
        strategy_type = self.comboBox.currentText()
        all_data1.oppo_policy = strategy_type

        battle_mode = self.comboBox_2.currentText()
        all_data1.battle_mode = battle_mode

        # 更新数据
        all_data1.hangcheng = all_data1.cac_hangcheng(all_data1.oppo_action)
        all_data1.oil_cost = all_data1.cac_oil(all_data1.agent_num)

        # 在终端显示
        self.print_to_terminal("==========================")
        self.print_to_terminal(f"对手数量：{adversary_num}")
        self.print_to_terminal(f"我方数量：{agent_num}")
        self.print_to_terminal(f"对手策略类型：{strategy_type}")
        self.print_to_terminal(f"对战模式：{battle_mode}")
        self.print_to_terminal("==========================")


        # 更新UI显示
        self.update_display_from_data()

        # 在窗口内显示图片
        self.display_image(battle_mode)

    def display_image(self, mode):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            if mode == "空对空":
                # 加载并显示空对空模式的PNG图像
                img = mpimg.imread('C:/Users\ASUS\Desktop/bishe\MBAM\MBAM/UI_test/kdk.png')
                ax.imshow(img)
                #ax.set_title("空对空模式", fontproperties=self.zh_font)
            elif mode == "空对地":
                # 加载并显示空对地模式的PNG图像
                img = mpimg.imread('C:/Users\ASUS\Desktop/bishe\MBAM\MBAM/UI_test/kdd.png')
                ax.imshow(img)
                #ax.set_title("空对地模式", fontproperties=self.zh_font)

            self.canvas.draw()
        except Exception as e:
            print(f"Error displaying image: {e}")

    def print_to_terminal(self, text):
        self.terminal_output.appendPlainText(text)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "仿真推演"))
        self.label.setText(_translate("MainWindow", "对手类型："))
        self.label_2.setText(_translate("MainWindow", "我方数量："))
        self.label_3.setText(_translate("MainWindow", "对手数量："))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "结束"))
        self.pushButton_3.setText(_translate("MainWindow", "载入预设"))
        self.pushButton_4.setText(_translate("MainWindow", "推演"))
        self.label_4.setText(_translate("MainWindow", "预设"))
        self.label_5.setText(_translate("MainWindow", "对战模式："))
        self.label_6.setText(_translate("MainWindow", "航程："))
        self.label_7.setText(_translate("MainWindow", "燃油："))
        self.label_8.setText(_translate("MainWindow", "对手编号："))


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,args):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.args = args
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.pushButton.clicked.connect(self.start_simulation)
        self.pushButton_2.clicked.connect(self.stop_simulation)
        #self.env = make_env('simple_tag_v6',in_contain_done=False)
        self.env = Simple_Tag()
        self.observations = self.env.reset()

    def start_simulation(self):
        self.print_to_terminal("仿真开始···")
        self.timer.start(100)

    def stop_simulation(self):
        self.print_to_terminal("仿真结束！")
        self.timer.stop()

    def update_simulation(self):

        try:

            #随机步self.observations, reward, done, info = self.env.step(
            #    [self.env.action_space.sample() for _ in self.env.agents])
            file_dir = "D:/document/MBAM/data/PPO_6v6/"
            player1_file = file_dir + "PPO_MH_player1__player1_iter78200.ckp"
            player2_file = file_dir + "PPO_MH_player2_player2_iter78200.ckp"
            '''
            player1_type = "ppo_mh" # "mbam_mh_om_mh"
            player2_type = "mbam_om_mh"
            

            player1_ctor = None
            player2_ctor = None
            '''
            agent1 = PPO_MH.load_model(filepath=player1_file, args=self.args, logger=None, device=self.args.device)
            #agent2 = MBAM_OM_MH.load_model(filepath=player2_file, args=self.args, logger=None, device=self.args.device, env_model=None)
            agent2 = PPO_MH.load_model(filepath=player2_file, args=self.args, logger=None, device=self.args.device)
            env = Simple_Tag()
            #agents = [agent1, agent2]

            for i in range(100):
                dis = 0
                s = env.reset()
            while True:
                time.sleep(0.02)
                frame = self.env.render("mode=rgb_array")
                #print(type(frame))
                frame=frame[0]
                self.display_frame(frame)
                #env.render()
                #print(type(frame))
                action_info1 = []
                for i in range (10):
                    action_info1.append(agent1.choose_action(state=s[i][0]))
                oppo_a = []
                for i in range (10):
                    oppo_a.append([a.item() for a in action_info1[i][0]])
                
                #源代码a = agent2.choose_action(state=s[1], oppo_hidden_prob=action_info1[5])[0].item()
                start = time.clock()
                action_info2 = []
                for i in range (10):
                    action_info2.append(agent2.choose_action(state=s[i][1]))
                end = time.clock()
                a = []
                for i in range (10):
                    a.append([a.item() for a in action_info2[i][0]])      
                oppo_action = []
                arrow_base = 0x2190 
                '''
                for i in range(6):
                    if oppo_a[i] == 0:
                        oppo_action.append("-")
                    else: 
                        offset = (oppo_a[i] - 1) % 4
                        oppo_action.append(chr(arrow_base+offset))
                '''
                
                
                actions = [oppo_a,a]

                s_, rew, done, _ = env.step(actions)
                s = s_
                if done:
                        break

                #while True:
                '''
                time.sleep(0.2)
                #env.render()
                frame = self.env.render("mode=rgb_array")
                #print(type(frame))
                frame=frame[0]
                
                self.display_frame(frame)
                action_info1 = []
                for i in range (10):
                    action_info1.append(agent1.choose_action(state=s[i][0]))
                oppo_a = []
                for i in range (10):
                    oppo_a.append([a.item() for a in action_info1[i][0]])
                #源代码a = agent2.choose_action(state=s[1], oppo_hidden_prob=action_info1[5])[0].item()
                start = time.clock()
                action_info2 = []
                for i in range (10):
                    action_info2.append(agent2.choose_action(state=s[i][1]))
                end = time.clock()
                a = []
                for i in range (10):
                    a.append([a.item() for a in action_info2[i][0]])      
                oppo_action = []
                arrow_base = 0x2190 
                for i in range(6):
                    if oppo_a[i] == 0:
                        oppo_action.append("-")
                    else: 
                        offset = (oppo_a[i] - 1) % 4
                        oppo_action.append(chr(arrow_base+offset))
                
                actions = []
                for i in range(10):
                    actions.append(oppo_a[i])
                for i in range(10):
                    actions.append(a[i])   
                s_, rew, done, _ = self.env.step(actions)
                
                for i in range (6):
                    if a[i] != 0:
                    dis += 1
                all_data1.oppo_action = oppo_action
                all_data1.hangcheng = str(dis)   
                all_data1.cac_time =  end - start
                self.update_display_from_data()  
                all_data1.our_shots = dis
                all_data1.their_shots = dis   
                ''' 


        except Exception as e:
            self.print_to_terminal(f"Error during simulation update: {e}")

    def display_frame(self, frame):

        try:
            # 检查并调整图像数据的形状

            if len(frame.shape) == 4 and frame.shape[0] == 1:
                frame = frame[0]
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.axis("off")
            ax.imshow(frame)
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            self.canvas.draw()
        except Exception as e:
            self.print_to_terminal(f"Error displaying frame: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--num_trj", type=int, default=3, help="Number of trajectories")
    parser.add_argument("--num_om_layers", type=int, default=5, help="Number of trajectories")
    parser.add_argument("--device", type=str, default="cpu", help="")
    parser.add_argument("--actor_rnn", type=bool, default=False, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=1, help="")
    parser.add_argument("--save_per_epoch", type=int, default=5, help="")
    parser.add_argument("--true_prob", type=bool, default=True, help="True or False, edit Actor_RNN.py line 47-48")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--only_use_last_layer_IOP", type=bool, default=False, help="True or False")
    parser.add_argument("--random_best_response", type=bool, default=False, help="True or False")
    parser.add_argument("--record_more", type=bool, default=False, help="True or False")
    parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    mainWindow = MainWindow(args=args)
    mainWindow.show()
    try:
        sys.exit(app.exec_())
    except SystemExit as e:
        print(f"System exit: {e}")
    except Exception as e:
        print(f"Exception in main thread: {e}")
