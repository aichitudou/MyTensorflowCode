# -*- coding: utf-8 -*-
import tensorflow as tf

#
# #############1. 构造图阶段###########################
# # 创建一个常量op，产生一个1*2矩阵，这个op被作为一个节点，加到默认图中
#
# # 构造器的返回值代表该常量op的返回值
# matrix1 = tf.constant([[3., 3.]])
#
# # 创建另外一个常量op， 产生一个2*1矩阵
# matrix2 = tf.constant([[2.], [2.]])
#
# # 创建一个矩阵乘法 matmul op， 把'matrix1'  和 'matrix2' 作为输入
# # 返回值 'product' 代表矩阵乘法的结果
# product = tf.matmul(matrix1, matrix2)
#
# #################2. 启动图阶段###########################
# # 启动默认图
# sess = tf.Session()
#
# #　调用sess 的run() 方法来执行矩阵乘法op， 传入'product' 作为该方法的参数
# # 上面提到， 'product' 代表了矩阵乘法 op 的输出，传入它是向方法表明，我们希望
# # 取回矩阵乘法op的输出
# #
# # 整个执行过程是自动化的，会话负责传递op所需的全部输入， op通常是并发执行的
# #　
# # 函数调用'run(product)'触发了图中三个op（两个常量op和一个矩阵乘法op）的执行
# #
# # 返回值'result'是一个numpy'ndarray'对象
#
# # result = sess.run(product)
# # print result
#
# # 任务完成，关闭会话
# # sess.close()
#
# # Session对象在使用完后需要关闭以释放资源，除了显式调用close外，也可以使用"with"
# # 代码块来自动完成关闭动作。
# with tf.Session() as sess:
#     result = sess.run([product])
#     print result
#

#
# # 创建一个变量，初始化为标量 0
# state = tf.Variable(0, name="counter")
#
# # 创建一个op，其作用是使state增加1
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update =tf.assign(state, new_value)
#
# # 启动图后，变量必须先经过'初始化'(init)op 初始化
# # 首先必须增加一个'初始化'op到图中
#
# init_op = tf.initialize_all_variables()
#
# # 启动图，运行op
# with tf.Session() as sess:
#     # 运行 'init' op
#     sess.run(init_op)
#     # 打钱 'state'的初始值
#     print sess.run(state)
#     for _ in range(3):
#         sess.run(update)
#         print sess.run(state)

#
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.mul(input1, intermed)
#
# with tf.Session() as sess:
#     result = sess.run([mul, intermed])
#     print result
#
# # 输出:
# # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
#


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

















