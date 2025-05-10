function draw() {
    background(240); // 设置背景颜色为浅灰色

    frameCounter++; // 增加帧计数器

    // 绘制地标（最底层）
    for (let landmark of landmarks) {
        fill(231, 76, 60); // 设置地标颜色为红色
        ellipse(landmark.position.x, landmark.position.y, 15, 15); // 绘制地标为圆形
    }

    // 更新并绘制代理（覆盖在地标之上）
    for (let agent of agents) {
        // 每隔指定帧数才决策一次动作
        if (frameCounter % decisionInterval === 0) {
            const action = decideAction(agent.observationState); // 根据观测状态决策动作
            stepEnvironment(agent, action); // 执行动作并更新环境
        } else {
            stepEnvironment(agent, agent.actionState); // 保持当前动作状态
        }

        // 绘制代理
        fill(52, 152, 219); // 设置代理颜色为蓝色
        ellipse(agent.position.x, agent.position.y, 20, 20); // 绘制代理为圆形
    }
}
