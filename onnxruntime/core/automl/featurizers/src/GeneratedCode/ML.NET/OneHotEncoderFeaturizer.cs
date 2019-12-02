// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(OneHotEncoderTransformer), null, typeof(SignatureLoadModel),
    OneHotEncoderTransformer.UserName, OneHotEncoderTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(OneHotEncoderTransformer), null, typeof(SignatureLoadRowMapper),
OneHotEncoderTransformer.UserName, OneHotEncoderTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(OneHotEncoderEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class OneHotEncoderExtensionClass
    {
        public static OneHotEncoderEstimator OneHotEncoderTransformer(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null /* Insert additional params here as needed*/)
        {
            var options = new OneHotEncoderEstimator.Options
            {
                Columns = new OneHotEncoderEstimator.Column[] { new OneHotEncoderEstimator.Column() { Name = outputColumnName, Source = inputColumnName ?? outputColumnName } },
                
                /* Codegen: add extra options here as needed */
            };

            return new OneHotEncoderEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }

        public static OneHotEncoderEstimator OneHotEncoderTransformer(this TransformsCatalog catalog, InputOutputColumnPair[] columns /* Insert additional params here as needed*/)
        {
            var options = new OneHotEncoderEstimator.Options
            {
                Columns = columns.Select(x => new OneHotEncoderEstimator.Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray(),
                
                /* Codegen: add extra options here as needed */
            };

            return new OneHotEncoderEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    public class OneHotEncoderEstimator : IEstimator<OneHotEncoderTransformer>
    {
        private Options _options;
        private readonly IHost _host;

        /* Codegen: Add additional needed class members here */

        #region Options

        /* If not one to one need to change this */
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options: TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            

            /* Codegen: Add additonal options as needed */
        }

        #endregion

        internal OneHotEncoderEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(OneHotEncoderEstimator));
            Contracts.CheckNonEmpty(options.Columns, nameof(options.Columns));
            /* Codegen: Any other checks for options go here */

            _options = options;
        }

        public OneHotEncoderTransformer Fit(IDataView input)
        {
            return new OneHotEncoderTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (var column in _options.Columns)
            {
                var inputColumn = columns[column.Source];

                if (!OneHotEncoderTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                    throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType.ToString()} for column {column.Name} not a supported type.");

                /* Codegen: Do correct schema mapping here */

            }
            return new SchemaShape(columns.Values);
        }
    }

    public sealed class OneHotEncoderTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = ""; /* Insert summary here */
        internal const string UserName = "OneHotEncoderTransformer";
        internal const string ShortName = "OneHotEncoderTransformer";
        internal const string LoadName = "OneHotEncoderTransformer";
        internal const string LoaderSignature = "OneHotEncoderTransformer";

        private TypedColumn[] _columns;
        private OneHotEncoderEstimator.Options _options;

        #endregion

        internal OneHotEncoderTransformer(IHostEnvironment host, IDataView input, OneHotEncoderEstimator.Options options) :
            base(host.Register(nameof(OneHotEncoderTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _columns = options.Columns.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString(), this)).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal OneHotEncoderTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(OneHotEncoderTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            var columnCount = ctx.Reader.ReadInt32();

            _options = new OneHotEncoderEstimator.Options();
            /* Codegen: Load any additional Options members here */

            _columns = new TypedColumn[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                _columns[i] = TypedColumn.CreateTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString(), ctx.Reader.ReadString(), this);

                // Load the C++ state and create the C++ transformer.
                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);
            }
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new OneHotEncoderTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            /* Codegen: Change these as needed */
            return new VersionInfo(
                modelSignature: "Enter 8 character long name here", /* Codegen: Enter * character name here */
                verWrittenCur: 0x00010001, /* Codegen: Update version numbers as necessary */
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OneHotEncoderTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            ctx.Writer.Write(_columns.Count());

            /* Codegen: Write any _options members needed here */

            foreach (var column in _columns)
            {
                ctx.Writer.Write(column.Name);
                ctx.Writer.Write(column.Source);
                ctx.Writer.Write(column.Type);

                // Save C++ state
                var data = column.CreateTransformerSaveData();
                ctx.Writer.Write(data.Length);
                ctx.Writer.Write(data);
            }
        }

        public void Dispose()
        {
            foreach (var column in _columns)
            {
                column.Dispose();
            }
        }

        #region ColumnInfo

        #region BaseClass

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Name;
            internal readonly string Source;
            internal readonly string Type;

            /* Codegen: Fill in supported types */
            private static readonly Type[] _supportedTypes = new Type[] {  };

            /* Codegen: Any other needed members */

            internal TypedColumn(string name, string source, string type)
            {
                Name = name;
                Source = source;
                Type = type;
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            public abstract void Dispose();

            public abstract Type ReturnType();

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
                {
                    byte[] savedData = new byte[bufferSize.ToInt32()];
                    Marshal.Copy(buffer, savedData, 0, savedData.Length);
                    return savedData;
                }
            }

            internal unsafe void CreateTransformerFromSavedData(byte[] data)
            {
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
            }

            internal static TypedColumn CreateTypedColumn(string name, string source, string type, OneHotEncoderTransformer parent)
            {
                
                throw new InvalidOperationException($"Column {name} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            internal TypedColumn(string name, string source, string type) :
                base(name, source, type)
            {
            }

            internal abstract TOutputType Transform(TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, TSourceType input, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool IsTrainingComplete(TransformerEstimatorSafeHandle estimatorHandle);
            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(IDataView input)
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    if (!IsTrainingComplete(estimatorHandle))
                    {
                        var fitResult = FitResult.Continue;
                        while (fitResult != FitResult.Complete)
                        {
                            fitResult = FitResult.Continue;
                            using (var data = input.GetColumn<TSourceType>(Source).GetEnumerator())
                            {
                                while (fitResult == FitResult.Continue && data.MoveNext())
                                {
                                    {
                                        success = FitHelper(estimatorHandle, data.Current, out fitResult, out errorHandle);
                                        if (!success)
                                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                                    }
                                }

                                success = CompleteTrainingHelper(estimatorHandle, out fitResult, out errorHandle);
                                if (!success)
                                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                            }
                        }
                    }
                    success = CreateTransformerFromEstimatorHelper(estimatorHandle, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }
        }

        #endregion
        
        #endregion

        private sealed class Mapper : MapperBase
        {
            #region Class members

            private readonly OneHotEncoderTransformer _parent;
            /* Codegen: add any extra class members here */

            #endregion

            public Mapper(OneHotEncoderTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(x.ReturnType()))).ToArray();
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                ValueGetter<TOutputType> result = (ref TOutputType dst) =>
                {
                    var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                    var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                    TSourceType value = default;
                    srcGetterScalar(ref value);

                    dst = ((TypedColumn<TSourceType, TOutputType>)_parent._columns[iinfo]).Transform(value);

                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Type inputType = input.Schema[_parent._columns[iinfo].Source].Type.RawType;
                Type outputType = _parent._columns[iinfo].ReturnType();

                return Utils.MarshalInvoke(MakeGetter<int, int>, new Type[] { inputType, outputType }, input, iinfo);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < InputSchema.Count; i++)
                {
                    if (_parent._columns.Any(x => x.Source == InputSchema[i].Name))
                    {
                        active[i] = true;
                    }
                }

                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    internal static class OneHotEncoderEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.OneHotEncoder",
            Desc = OneHotEncoderTransformer.Summary,
            UserName = OneHotEncoderTransformer.UserName,
            ShortName = OneHotEncoderTransformer.ShortName)]
        public static CommonOutputs.TransformOutput OneHotEncoder(IHostEnvironment env, OneHotEncoderEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, OneHotEncoderTransformer.ShortName, input);
            var xf = new OneHotEncoderEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
