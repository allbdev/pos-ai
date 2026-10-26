// ====================================================================
// 🩹 tf: importa o TensorFlow.js (Node) com uma correção de compatibilidade
// ====================================================================
// O @tensorflow/tfjs-node 4.22 usa util.isNullOrUndefined, uma função
// do Node que foi REMOVIDA no Node 23. No Node 24, qualquer operação
// (oneHot, fit, predict...) quebra com:
//   TypeError: (0 , util_1.isNullOrUndefined) is not a function
//
// A correção é recolocar a função antes de usar o tf.
// Funciona porque o tfjs-node procura util.isNullOrUndefined na hora da chamada.
//
// Por isso os outros arquivos importam o tf DAQUI, e não direto do pacote:
//   import tf from './tf.js'
//
// Alternativa: usar o Node 22 (LTS), que ainda tem a função.

import util from 'node:util'
import tf from '@tensorflow/tfjs-node'

// ??= só define se ainda não existir (no Node 22 a original é mantida)
util.isNullOrUndefined ??= value => value === null || value === undefined

export default tf
