import { buildLayout } from "./layout";

export default async function main(game) {
    const container = buildLayout(game.app);
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

    game.stage.aim.visible = false;

    worker.onmessage = async ({ data }) => {
        const { type, x, y } = data;

        if (type === 'prediction') {
            console.log(`🎯 AI predicted at: (${x}, ${y})`);
            container.updateHUD(data);
            game.stage.aim.visible = true;

            game.stage.aim.setPosition(data.x, data.y);
            const position = game.stage.aim.getGlobalPosition();

            game.handleClick({
                global: position,
            });

        }

        if (type === 'done') {
            // Limita a frequencia de tiros
            setTimeout(async () => {
                // Envia o próximo frame
                const bitmap = await getframe(game);

                worker.postMessage({
                    type: 'predict',
                    image: bitmap,
                }, [bitmap]);
            }, 100)
        }

    };

    // Envia o frame ao iniciar o projeto
    const bitmap = await getframe(game);

    worker.postMessage({
        type: 'predict',
        image: bitmap,
    }, [bitmap]);


    return container;
}

const getframe = async (game) => {
    const canvas = game.app.renderer.extract.canvas(game.stage);
    const bitmap = await createImageBitmap(canvas);
    return bitmap;
}