console.log('this is a test');

let VIEWS = {'Age': 'Customer_Age_view.png', 'Gender': 'Gender_view.png', 'Dependents': 'Dependent_count_view.png', 'Education Level': 'Education_Level_view.png', 'Marital Status': 'Marital_Status_view.png', 'Annual Income': 'Income_Category_view.png'};

function changeView(demo) {
    console.log(`static/assets/${VIEWS[demo]}`);
    d3.selectAll('#demoView').attr('src', `static/assets/${VIEWS[demo]}`)
};