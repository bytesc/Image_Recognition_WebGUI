import PIL.Image
import torch.nn
import random
from io import BytesIO
from hot_img import hot_img

import pywebio


from model import *
import nibabel as nib
import data.datasets
import time

from pyecharts.charts import Bar, Page
from pyecharts.globals import ThemeType
from pyecharts import options as opts

# from pyecharts.globals import CurrentConfig, OnlineHostType
# CurrentConfig.ONLINE_HOST = "../pyecharts-assets-master/assets/"
# from pyecharts.globals import WarningType
# WarningType.ShowWarning = False


def bar_base(data) -> Bar:
    c = (
        Bar({"theme": ThemeType.MACARONS})
        .add_xaxis(["CN", "EMCI", "MCI", "LMCI", "AD"])
        .add_yaxis("output_value", data, markpoint_opts=["max"])
        .set_global_opts(
            title_opts={"text": "模型输出", "subtext": ""},)
        )
    return c


def refresh():
    pywebio.output.clear()
    page1()


def generate_random_str(target_length=32):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(target_length):
        random_str += base_str[random.randint(0, length)]
    return random_str


def make_image_list(path, user_ip, hot_type="LRP"):
    img_list = []
    x = 10
    for i in range(1, x):
        path_group = hot_img(path, 64 + 16 * i, hot_type)
        img_list.append(list(path_group))
    print_logs("make_img," + str(img_list[1:][0]).strip("[").strip("]")+",\n", user_ip)
    return img_list


def show_img_s(path, user_ip, mod, hot_type="LRP"):
    pywebio.output.popup("图像渲染可能花费很长时间，请耐心等待", [pywebio.output.put_row(
        [pywebio.output.put_loading(shape="grow", color="success")],
    )])
    path_group = list(hot_img(path, 64 + 16 * 7, hot_type))
    print_logs("make_img," + str(path_group[0]).strip("[").strip("]") + ",\n", user_ip)
    img_table = None
    if mod == 3:
        for j in range(3):
            path_group[j] = PIL.Image.open(path_group[j])
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(path_group[2])],
        ])
    if mod == 1:
        for j in range(3):
            path_group[j] = PIL.Image.open(path_group[j])
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(path_group[0])],
        ])
    if mod == 2:
        for j in range(3):
            path_group[j] = PIL.Image.open(path_group[j])
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(path_group[1])],
        ])
    return pywebio.output.popup(title='图像', content=img_table)


def show_img(path, user_ip, mod):
    pywebio.output.popup("图像渲染可能花费很长时间，请耐心等待", [pywebio.output.put_row(
        [pywebio.output.put_loading(shape="grow", color="success")],
    )])
    img_list = make_image_list(path, user_ip)
    for i in img_list:
        for j in range(3):
            i[j] = PIL.Image.open(i[j])

    img_table = []
    if mod == 1:
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(img_list[8][0]), pywebio.output.put_image(img_list[7][0]),
             pywebio.output.put_image(img_list[6][0])],
            [pywebio.output.put_image(img_list[5][0]), pywebio.output.put_image(img_list[4][0]),
             pywebio.output.put_image(img_list[3][0])],
            [pywebio.output.put_image(img_list[2][0]), pywebio.output.put_image(img_list[1][0]),
             pywebio.output.put_image(img_list[0][0])],
        ])

    if mod == 2:
        img_table = pywebio.output.put_table([

            [pywebio.output.put_image(img_list[8][1]), pywebio.output.put_image(img_list[7][1]),
             pywebio.output.put_image(img_list[6][1])],
            [pywebio.output.put_image(img_list[5][1]), pywebio.output.put_image(img_list[4][1]),
             pywebio.output.put_image(img_list[3][1])],
            [pywebio.output.put_image(img_list[2][1]), pywebio.output.put_image(img_list[1][1]),
             pywebio.output.put_image(img_list[0][1])],

        ])
    if mod == 3:
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(img_list[8][2]), pywebio.output.put_image(img_list[7][2]),
             pywebio.output.put_image(img_list[6][2])],
            [pywebio.output.put_image(img_list[5][2]), pywebio.output.put_image(img_list[4][2]),
             pywebio.output.put_image(img_list[3][2])],
            [pywebio.output.put_image(img_list[2][2]), pywebio.output.put_image(img_list[1][2]),
             pywebio.output.put_image(img_list[0][2])]
        ])
    return pywebio.output.popup(title='图像', content=img_table)


def compare_ans(a, ans):
    if a == ans:
        return True
    else:
        return False


def print_logs(content, user_ip):
    with open("./run_logs/" + user_ip+"run_logs.csv", 'a') as file:
        file.write(str(pywebio.session.info.user_ip) + "," +
                   str(pywebio.session.info.user_agent.device.model) + "," +
                   str(pywebio.session.info.user_agent.browser.family) + "," + time.ctime() + ",")
        file.write(content)


@pywebio.config(title="Demo", description="基于ADNI数据集的阿尔兹海默病诊断",)
def page1(is_demo=False):
    user_ip = str(pywebio.session.info.user_ip)+generate_random_str(16)
    ans = "认知正常(CN)"
    ans_list = ["阿尔茨海默病(AD)", "认知正常(CN)", "轻度认知障碍(MCI)", "早期轻度认知障碍(EMCI)",
                "晚期轻度认知障碍(LMCI)"]

    ans_y = [1, 0, 0, 0, 0]

    chart_html = bar_base(ans_y).render_notebook()

    temp_file_path = "demo.nii"

    graph_img = PIL.Image.open("./data/net_graph.png")
    # front_img = PIL.Image.open("./data/front_page1.png")
    train_img = PIL.Image.open("./data/train_process2.png")
    brain_img = PIL.Image.open("./data/brain_demo.png")

    hot_img9 = PIL.Image.open("./data/hot_img9.3.png")
    hot_img1 = PIL.Image.open("./data/hot_img1.3.png")
    # hot_only = PIL.Image.open("./data/hot_only.png")
    brain_demo1 = PIL.Image.open("./data/brain_demo1.png")

    while 1:
        try:
            pywebio.output.put_warning("识别结果仅供参考", closable=True, position=- 1)

            pywebio.output.put_html("<h1><center>基于ADNI数据集的阿尔兹海默病诊断</center></h1><hr>")

            cn_content = [pywebio.output.put_markdown("认知正常")]
            emci_content = [pywebio.output.put_markdown("早期轻度认知障碍")]
            mci_content = [pywebio.output.put_markdown("轻度认知障碍")]
            lmci_content = [pywebio.output.put_markdown("晚期轻度认知障碍")]
            ad_content = [pywebio.output.put_markdown("阿尔茨海默病")]

            pywebio.output.put_row(
                [pywebio.output.put_scope(name="chart", content=[pywebio.output.put_html(chart_html)])
                 ],
            )

            pywebio.output.put_row(
                [pywebio.output.put_collapse("CN", cn_content, open=compare_ans("认知正常(CN)", ans)),
                 pywebio.output.put_collapse("EMCI", emci_content, open=compare_ans("早期轻度认知障碍(EMCI)", ans)),
                 pywebio.output.put_collapse("MCI", mci_content, open=compare_ans("轻度认知障碍(MCI)", ans)),
                 pywebio.output.put_collapse("LMCI", lmci_content, open=compare_ans("晚期轻度认知障碍(LMCI)", ans)),
                 pywebio.output.put_collapse("AD", ad_content, open=compare_ans("阿尔茨海默病(AD)", ans))],
            )

            more_content = [
                pywebio.output.put_table([
                    [
                        pywebio.output.put_image(hot_img9),
                        pywebio.output.put_image(hot_img1),
                    ],
                    [
                        pywebio.output.put_image(brain_img),
                        pywebio.output.put_image(brain_demo1),
                    ],
                ])
            ]
            f = open("model.py", "r", encoding="UTF-8")
            code = f.read()
            f.close()
            pywebio.output.put_collapse("热力图demo", more_content, open=True, position=- 1)
            pywebio.output.put_row([
                pywebio.output.put_collapse("模型信息", [pywebio.output.put_image(graph_img)], open=True, position=- 1),
                pywebio.output.put_collapse("训练信息", [pywebio.output.put_image(train_img),
                                                         pywebio.output.put_markdown(
                                                             "learning_rate=1e-4 weight_decay=1e-5"),
                                                         pywebio.output.put_markdown("batch_size=4 num_works=1"), ],

                                            open=True, position=- 1)
            ])
            pywebio.output.put_collapse("模型代码", [pywebio.output.put_code(code, "python")], open=False, position=- 1)
            pywebio.output.put_markdown("ref: https://github.com/moboehle/Pytorch-LRP")
            pywebio.output.put_markdown("datasets: https://adni.loni.usc.edu")

            action = pywebio.input.actions(' ',
                                           [{'label': "上传.nii图像", 'value': "上传.nii图像", 'color': 'warning'},
                                            {'label': "使用demo.nii", 'value': "使用demo.nii", 'color': 'info'},
                                            "查看图像"
                                            ])
            if action == "使用demo.nii":
                is_demo = True
            if action == "上传.nii图像":
                is_demo = False
            if action == "查看图像":
                action = pywebio.input.actions(' ',
                                               ["查看原图",
                                                "查看热力图",
                                                "查看单层原图",
                                                "查看单层热力图",
                                                {'label': "自定义查看", 'value': "自定义查看", 'color': 'dark',
                                                 "disabled": True}
                                                ])
                if action == "查看原图":
                    # pywebio.output.popup("功能不可用",
                    #                      [pywebio.output.put_markdown("由于渲染消耗算力极大，此功能在CPU服务器不可用"),
                    #                       pywebio.output.put_markdown("### 请点击“查看单层原图”")])
                    show_img(temp_file_path, user_ip, 1)
                    pywebio.output.clear()
                    continue
                if action == "查看热力图":
                    # pywebio.output.popup("功能不可用",
                    #                      [pywebio.output.put_markdown("由于渲染消耗算力极大，此功能在CPU服务器不可用"),
                    #                       pywebio.output.put_markdown("### 请点击“查看单层热力图”")])
                    show_img(temp_file_path, user_ip, 3)
                    pywebio.output.clear()
                    continue
                if action == "查看单层原图":
                    show_img_s(temp_file_path, user_ip, mod=1)
                    pywebio.output.clear()
                    continue
                if action == "查看单层热力图":
                    show_img_s(temp_file_path, user_ip, mod=3)
                    pywebio.output.clear()
                    continue

            ###################################################################################

            if is_demo is False:
                try:
                    inpic = pywebio.input.file_upload(label="上传医学影像(.nii)")
                    inpic = BytesIO(inpic['content'])
                    temp_file_path = "./nii/" + generate_random_str() + ".nii"
                    with open(temp_file_path, 'wb') as file:
                        file.write(inpic.getvalue())  # 保存到本地
                    print_logs("upload_file," + temp_file_path + ",\n", user_ip)
                except:
                    pywebio.output.toast("输入错误，请上传医学影像文件(.nii)", color="warn")
                    refresh()

            if is_demo is True:
                is_demo = False
                temp_file_path = "demo.nii"

            pywebio.output.popup("AI识别中", [pywebio.output.put_row(
                [pywebio.output.put_loading(shape="grow", color="success")],
            )])

            ##############################################################################

            torch.no_grad()
            test_model = torch.load("./data/model_save/myModel_130.pth", map_location=torch.device('cpu'))
            test_model.eval()
            # print(test_model)

            img = None
            try:
                img = nib.load(temp_file_path)
                img = img.get_fdata()
                img = data.datasets.process_img(img)
                img = img.reshape((1, 1, -1, 256, 256))
                # print(img.shape)
            except Exception:
                pywebio.output.toast("输入处理错误，请上传医学影像文件(.nii)\t应大于：(168x168x168)", color="warn")
                refresh()

            try:
                output = None
                with torch.no_grad():
                    output = test_model(img)
                ans_y = output.squeeze().tolist()
            except Exception:
                pywebio.output.toast("模型识别错误，可能由于服务器内存不足，请稍后重试", color="warn")
                refresh()

            # print(output)
            if min(ans_y) < 0:
                m = min(ans_y)
                for i in range(len(ans_y)):
                    ans_y[i] -= 1.2 * m
            ans = ans_list[output.argmax(1).item()]
            # print(ans)

            ######################################################################

            chart_html = bar_base([ans_y[1], ans_y[3], ans_y[2], ans_y[4], ans_y[0]]).render_notebook()
            with pywebio.output.use_scope(name="chart") as scope_name:
                pywebio.output.clear()
                pywebio.output.put_html(chart_html)
            # print(chart_html)

            show_result = [pywebio.output.put_markdown("诊断为：\n # " + ans)]
            pywebio.output.popup(title='AI识别结果', content=show_result)

            pywebio.output.clear()

        except Exception:
            continue


if __name__ == "__main__":
    # page1()
    pywebio.platform.start_server(
        applications=[page1, ],
        debug=False,
        auto_open_webbrowser=False,
        remote_access=False,
        cdn=False,
        port=6006
    )

