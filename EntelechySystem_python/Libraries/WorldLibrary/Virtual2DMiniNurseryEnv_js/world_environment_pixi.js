// import * as PIXI from 'pixi.js';

// 创建 PIXI 应用
const app = new PIXI.Application({
    width: 800,
    height: 600,
    backgroundColor: 0xffffff,
});
document.body.appendChild(app.canvas);

// 初始化全局变量
const agents = [];
const landmarks = [];
const agentGraphics = [];
const landmarkGraphics = [];

// 创建代理人和地标
function setup() {
    createAgents();
    createLandmarks();
    app.ticker.add(update);
}

// 创建代理人
function createAgents() {
    for (let i = 0; i < 3; i++) {
        const agent = {
            x: Math.random() * 700 + 100,
            y: Math.random() * 500 + 100,
            vx: Math.random() * 2 - 1,
            vy: Math.random() * 2 - 1,
        };
        agents.push(agent);

        const circle = new PIXI.Graphics();
        circle.beginFill(0x3498db);
        circle.drawCircle(0, 0, 20);
        circle.endFill();
        circle.x = agent.x;
        circle.y = agent.y;
        app.stage.addChild(circle);
        agentGraphics.push(circle);
    }
}

// 创建地标
function createLandmarks() {
    for (let i = 0; i < 3; i++) {
        const landmark = {
            x: Math.random() * 700 + 100,
            y: Math.random() * 500 + 100,
        };
        landmarks.push(landmark);

        const circle = new PIXI.Graphics();
        circle.beginFill(0xe74c3c);
        circle.drawCircle(0, 0, 15);
        circle.endFill();
        circle.x = landmark.x;
        circle.y = landmark.y;
        app.stage.addChild(circle);
        landmarkGraphics.push(circle);
    }
}

// 更新代理人位置
function update() {
    agents.forEach((agent, index) => {
        agent.x += agent.vx;
        agent.y += agent.vy;

        // 边界检测
        if (agent.x < 0 || agent.x > 800) agent.vx *= -1;
        if (agent.y < 0 || agent.y > 600) agent.vy *= -1;

        // 更新图形位置
        agentGraphics[index].x = agent.x;
        agentGraphics[index].y = agent.y;
    });
}

// 启动应用
setup();

// // //HACK 一个最简单的示例代码。
// // Create the application helper and add its render target to the page
// const app = new PIXI.Application();
// await app.init({ width: 640, height: 360 })
// document.body.appendChild(app.canvas);

// // Create the sprite and add it to the stage
// await PIXI.Assets.load('./assets/star.png');
// let sprite = PIXI.Sprite.from('./assets/star.png');
// app.stage.addChild(sprite);

// // Add a ticker callback to move the sprite back and forth
// let elapsed = 50.0;
// app.ticker.add((ticker) => {
//     elapsed += ticker.deltaTime;
//     sprite.x = 100.0 + Math.cos(elapsed / 50.0) * 100.0;
// });

