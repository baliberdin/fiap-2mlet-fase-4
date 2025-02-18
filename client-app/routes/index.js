var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', async function(req, res, next) {
  res.render('index', { title: 'Previsão de preço de Ações' });
});

router.get('/data', async function(req, res, next) {
  response = await fetch(`http://${process.env.STOCK_API_HOST}:${process.env.STOCK_API_PORT}/?history_days=100&prediction_days=7`)
  stock_data = await response.json();
  real_price = {x:[], y:[]};
  pred_price = {x:[], y:[]};
  future_price = {x:[], y:[]};

  for( let i=0; i<stock_data.real_price.length; i++){
    real_price.x.push(stock_data.real_price[i]["date"]);
    real_price.y.push(stock_data.real_price[i]["close"].toFixed(2));
  }

  for( let i=0; i<stock_data.predicted_price.length; i++){
    pred_price.x.push(stock_data.predicted_price[i]["date"]);
    pred_price.y.push(stock_data.predicted_price[i]["close"].toFixed(2));
  }

  for( let i=0; i<stock_data.future_price.length; i++){
    future_price.x.push(stock_data.future_price[i]["date"]);
    future_price.y.push(stock_data.future_price[i]["close"].toFixed(2));
  }

  res.send({real_price:real_price, pred_price:pred_price, future_price: future_price});
});

module.exports = router;
