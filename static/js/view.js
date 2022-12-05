let VIEWS = {'Age': 'Customer_Age_view.png', 'Gender': 'Gender_view.png', 'Dependents': 'Dependent_count_view.png', 'Education Level': 'Education_Level_view.png', 'Marital Status': 'Marital_Status_view.png', 'Annual Income': 'Income_Category_view.png'};

function init() {
    let select = d3.select('#demo')
    for (i = 0; i < Object.keys(VIEWS).length; i++) {
        let demo = Object.keys(VIEWS)[i];
        select.append('option').attr('value', demo).text(demo)
    };
    let demo = d3.select('#demo').select('option').attr('value')
    console.log(demo)
    d3.select('#demoView').attr('src', `static/assets/${VIEWS[demo]}`);
};

function changeView(demo) {
    d3.select('#demoView').attr('src', `static/assets/${VIEWS[demo]}`);
};

init();