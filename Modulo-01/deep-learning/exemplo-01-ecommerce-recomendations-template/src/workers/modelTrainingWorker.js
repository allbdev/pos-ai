import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
const _globalCtx = {};

const WEIGHTS = {
    category: .4,
    color: .3,
    price: .2,
    age: .1
}

const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

/**
 * @param {Array<{id: number, name: string, category: string, price: number, color: string}>} catalog
 * @param {Array<{id: number, name: string, age: number, purchases: Array<{id: number, name: string, category: string, price: number, color: string}>}>} users
 */
function makeContext(catalog, users) {
    const ages = users.map(u => u.age)
    const prices = catalog.map(p => p.price)

    const maxAge = Math.max(...ages)
    const minAge = Math.min(...ages)

    const maxPrice = Math.max(...prices)
    const minPrice = Math.min(...prices)

    const colors = [...new Set(catalog.map(p => p.color))]
    const categories = [...new Set(catalog.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        })
    )

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        })
    )

    // Computar a média de idade dos compradores por produto ajuda a personalizar
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        catalog.map(product => {
            const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge
            const norm = normalize(avg, minAge, maxAge)

            return [product.name, norm]
        })
    )

    return {
        catalog,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        dimensions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weigth) => tf.oneHot(index, length).cast('float32').mul(weigth)

function encodeProduct(product, context) {
    // Normalizando dados para ficar de 0 a 1 e
    // aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const catalog = await (await fetch('/data/products.json')).json()

    const context = makeContext(catalog, users)

    context.productsVectors = catalog.map(product => ({
        name: product.name,
        meta: {
            ...product
        },
        vector: encodeProduct(product, context).dataSync()
    }))

    debugger

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
